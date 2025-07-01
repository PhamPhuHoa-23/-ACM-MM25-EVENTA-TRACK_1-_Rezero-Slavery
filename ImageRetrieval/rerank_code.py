import cv2
import json
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import logging
import argparse

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def load_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    query_mapping = {}
    for item in data:
        query_mapping[item['query_id']] = item
    
    return data, query_mapping

def get_common_queries(results_list):
    query_mappings = []
    for results, query_mapping in results_list:
        query_mappings.append(set(query_mapping.keys()))
    
    common_queries = query_mappings[0]
    for query_set in query_mappings[1:]:
        common_queries = common_queries.intersection(query_set)
    
    return sorted(list(common_queries))

def top2_pool_reranking(results_list, model_names, weights=None):
    if weights is None:
        weights = [1.0/len(results_list)] * len(results_list)
    
    query_mappings = [query_mapping for _, query_mapping in results_list]
    common_queries = get_common_queries(results_list)
    
    ensembled_results = []
    
    for query_id in common_queries:
        article_data = {}
        query_results = []
        
        for i, query_mapping in enumerate(query_mappings):
            if query_id in query_mapping:
                query_results.append(query_mapping[query_id])
                top2 = query_mapping[query_id]['results'][:2]
                
                for j, result in enumerate(top2):
                    article_id = result['article_id']
                    position_weight = 1.0 if j == 0 else 0.8
                    weighted_score = result['score'] * weights[i] * position_weight
                    
                    if article_id not in article_data:
                        article_data[article_id] = {
                            'image_id': result['image_id'],
                            'article_id': article_id,
                            'scores': [],
                            'models': []
                        }
                    
                    article_data[article_id]['scores'].append(weighted_score)
                    article_data[article_id]['models'].append(f"{model_names[i]}_rank{j+1}")
        
        if len(query_results) != len(results_list):
            continue
        
        pool_results = []
        for article_id, data in article_data.items():
            appearance_bonus = 0.03 * (len(data['scores']) - 1)
            final_score = np.mean(data['scores']) + appearance_bonus
            
            result = {
                'image_id': data['image_id'],
                'article_id': article_id,
                'score': final_score
            }
            pool_results.append(result)
        
        pool_results.sort(key=lambda x: x['score'], reverse=True)
        
        used_articles = {r['article_id'] for r in pool_results}
        remaining = []
        
        for query_result in query_results:
            for result in query_result['results']:
                if result['article_id'] not in used_articles:
                    clean_result = {
                        'image_id': result['image_id'],
                        'article_id': result['article_id'],
                        'score': result['score']
                    }
                    remaining.append(clean_result)
                    used_articles.add(result['article_id'])
        
        remaining.sort(key=lambda x: x['score'], reverse=True)
        final_results = (pool_results + remaining)[:10]
        
        ensembled_results.append({
            "query_id": query_id,
            "results": final_results
        })
    
    return ensembled_results

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def stage1_ensemble(model1_file, model2_file, model3_file, output_file, weights=None, model_names=None):
    try:
        model1_data, model1_mapping = load_results(model1_file)
        model2_data, model2_mapping = load_results(model2_file)
        model3_data, model3_mapping = load_results(model3_file)
        
        results_list = [(model1_data, model1_mapping), (model2_data, model2_mapping), (model3_data, model3_mapping)]
        
        if model_names is None:
            model_names = ['model1', 'model2', 'model3']
        
        common_queries = get_common_queries(results_list)
        
        print(f"Stage 1: {len(common_queries)} queries")
        
        if len(common_queries) == 0:
            print("No common queries found!")
            return None
        
        final_results = top2_pool_reranking(results_list, model_names, weights)
        
        save_results(final_results, output_file)
        print(f"Saved to {output_file}")
        return final_results
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"Stage 1 error: {e}")
        return None 

class EnhancedReranker:
    
    def __init__(self, query_images_path=None, database_images_path=None):
        self.confidence_threshold = 0.4
        self.min_inlier_ratio_diff = 0.2
        self.use_multiple_matchers = True
        
        self.query_images_path = query_images_path or "/mnt/e/Embedding/beit3_project/images/private_images"
        self.database_images_path = database_images_path or "/mnt/e/Embedding/beit3_project/images/database_images"
        
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.sift = cv2.SIFT_create(nfeatures=1000) if cv2.__version__ >= '4.4.0' else None
        
        self.stats = defaultdict(int)
        
    def get_image(self, image_id: str, folder: str) -> np.ndarray:
        if folder == "private_images" or folder == "query":
            base_path = self.query_images_path
        elif folder == "database_images" or folder == "database":
            base_path = self.database_images_path
        else:
            base_path = f"/mnt/e/Embedding/beit3_project/images/{folder}"
            
        path = f"{base_path}/{image_id}.jpg"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = cv2.imread(path)
        return img
    
    def compute_feature_matches(self, 
                              query_img: np.ndarray, 
                              candidate_img: np.ndarray,
                              method: str = 'orb'):
        
        gray1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY) if len(query_img.shape) == 3 else query_img
        gray2 = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY) if len(candidate_img.shape) == 3 else candidate_img
        
        results = {
            'inliers': 0,
            'inlier_ratio': 0.0,
            'homography_score': 0.0,
            'spatial_consistency': 0.0,
            'scale_consistency': 0.0
        }
        
        try:
            if method == 'orb':
                detector = self.orb
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                detector = self.sift
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return results
            
            matches = matcher.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    ratio_threshold = 0.7 if len(kp1) > 500 else 0.75
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4:
                return results
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(mask.sum())
                results['inliers'] = inliers
                results['inlier_ratio'] = inliers / len(good_matches)
                
                if homography is not None:
                    det = np.linalg.det(homography[:2, :2])
                    if 0.1 < det < 10:
                        results['homography_score'] = min(1.0, inliers / 50.0)
                    
                    if inliers > 0:
                        inlier_pts = src_pts[mask.ravel() == 1]
                        spatial_score = self._compute_spatial_distribution(inlier_pts, gray1.shape)
                        results['spatial_consistency'] = spatial_score
                    
                    scales = []
                    for i, m in enumerate(good_matches):
                        if mask[i]:
                            s1 = kp1[m.queryIdx].size
                            s2 = kp2[m.trainIdx].size
                            if s1 > 0 and s2 > 0:
                                scales.append(s2 / s1)
                    
                    if scales:
                        scale_std = np.std(scales)
                        results['scale_consistency'] = 1.0 / (1.0 + scale_std)
            
        except Exception:
            pass
        
        return results
    
    def _compute_spatial_distribution(self, points: np.ndarray, img_shape) -> float:
        if len(points) < 4:
            return 0.0
        
        grid_size = 4
        h, w = img_shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        grid_counts = np.zeros((grid_size, grid_size))
        for pt in points:
            x, y = pt
            grid_x = min(int(x // cell_w), grid_size - 1)
            grid_y = min(int(y // cell_h), grid_size - 1)
            grid_counts[grid_y, grid_x] += 1
        
        cells_with_points = np.sum(grid_counts > 0)
        distribution_score = cells_with_points / (grid_size * grid_size)
        
        if cells_with_points > 0:
            concentration = np.std(grid_counts[grid_counts > 0])
            distribution_score *= np.exp(-concentration / len(points))
        
        return distribution_score
    
    def compute_reranking_confidence(self, scores1, scores2, similarity_scores=None):
        
        score1 = (scores1['inliers'] * 0.4 + 
                 scores1['inlier_ratio'] * 100 * 0.3 +
                 scores1['homography_score'] * 50 * 0.2 +
                 scores1['spatial_consistency'] * 30 * 0.1)
        
        score2 = (scores2['inliers'] * 0.4 + 
                 scores2['inlier_ratio'] * 100 * 0.3 +
                 scores2['homography_score'] * 50 * 0.2 +
                 scores2['spatial_consistency'] * 30 * 0.1)
        
        confidence = 0.0
        
        inlier_diff = abs(scores1['inliers'] - scores2['inliers'])
        inlier_ratio_diff = abs(scores1['inlier_ratio'] - scores2['inlier_ratio'])
        
        if inlier_diff > 20:
            confidence += 0.4
        elif inlier_diff > 10:
            confidence += 0.2
        
        if inlier_ratio_diff > self.min_inlier_ratio_diff:
            confidence += 0.3
        
        min_inliers = min(scores1['inliers'], scores2['inliers'])
        if min_inliers > 15:
            confidence += 0.2
        elif min_inliers < 5:
            confidence -= 0.3
        
        spatial_diff = abs(scores1['spatial_consistency'] - scores2['spatial_consistency'])
        if spatial_diff > 0.3:
            confidence += 0.1
        
        if similarity_scores and len(similarity_scores) >= 2:
            sim_diff = abs(similarity_scores[0] - similarity_scores[1])
            if sim_diff < 0.01:
                confidence += 0.2
        
        should_rerank = confidence > self.confidence_threshold
        
        if max(scores1['inliers'], scores2['inliers']) < 8:
            should_rerank = False
            confidence = 0.0
        
        return should_rerank, confidence
    
    def rerank_with_multiple_features(self, query_img, candidate1_img, candidate2_img):
        
        orb_scores1 = self.compute_feature_matches(query_img, candidate1_img, 'orb')
        orb_scores2 = self.compute_feature_matches(query_img, candidate2_img, 'orb')
        
        final_scores1 = orb_scores1.copy()
        final_scores2 = orb_scores2.copy()
        
        if self.sift is not None:
            sift_scores1 = self.compute_feature_matches(query_img, candidate1_img, 'sift')
            sift_scores2 = self.compute_feature_matches(query_img, candidate2_img, 'sift')
            
            for key in final_scores1:
                if key in ['inliers']:
                    final_scores1[key] = max(orb_scores1[key], sift_scores1[key] * 0.8)
                    final_scores2[key] = max(orb_scores2[key], sift_scores2[key] * 0.8)
                else:
                    final_scores1[key] = 0.6 * orb_scores1[key] + 0.4 * sift_scores1[key]
                    final_scores2[key] = 0.6 * orb_scores2[key] + 0.4 * sift_scores2[key]
        
        return final_scores1, final_scores2
    
    def rerank_top2(self, json_path):
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        reranked = []
        
        for query in tqdm(data, desc="Stage 2"):
            query_id = query["query_id"]
            results = query["results"]
            
            similarity_scores = [r.get('score', 0) for r in results[:2]]
            
            try:
                query_img = self.get_image(query_id, "private_images")
                
                top2 = results[:2]
                others = results[2:]
                
                if len(top2) < 2:
                    reranked.append(query)
                    continue
                
                img1 = self.get_image(top2[0]["image_id"], "database_images")
                img2 = self.get_image(top2[1]["image_id"], "database_images")
                
                scores1, scores2 = self.rerank_with_multiple_features(query_img, img1, img2)
                
                should_rerank, confidence = self.compute_reranking_confidence(
                    scores1, scores2, similarity_scores
                )
                
                if should_rerank:
                    total_score1 = (scores1['inliers'] + 
                                  scores1['inlier_ratio'] * 100 + 
                                  scores1['homography_score'] * 50)
                    total_score2 = (scores2['inliers'] + 
                                  scores2['inlier_ratio'] * 100 + 
                                  scores2['homography_score'] * 50)
                    
                    if total_score2 > total_score1:
                        top2_sorted = [top2[1], top2[0]]
                        self.stats['swapped'] += 1
                    else:
                        top2_sorted = top2
                        self.stats['kept_after_check'] += 1
                else:
                    top2_sorted = top2
                    self.stats['skipped'] += 1
                
            except Exception:
                top2_sorted = results[:2]
                self.stats['errors'] += 1
            
            reranked.append({
                "query_id": query_id,
                "results": top2_sorted + others
            })
        
        return reranked

class OptimizedTop3Reranker:
    
    def __init__(self, 
                 confidence_threshold=0.2,
                 feature_quality_threshold=15,
                 enable_multiple_validations=True,
                 query_images_path=None,
                 database_images_path=None):
        
        self.confidence_threshold = confidence_threshold
        self.feature_quality_threshold = feature_quality_threshold
        self.enable_multiple_validations = enable_multiple_validations
        
        self.query_images_path = query_images_path or "/mnt/e/Embedding/beit3_project/images/private_images"
        self.database_images_path = database_images_path or "/mnt/e/Embedding/beit3_project/images/database_images"
        
        self.orb = cv2.ORB_create(
            nfeatures=2500,
            scaleFactor=1.15,
            nlevels=10,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        
        try:
            self.sift = cv2.SIFT_create(
                nfeatures=1200,
                nOctaveLayers=4,
                contrastThreshold=0.03,
                edgeThreshold=8,
                sigma=1.6
            )
            self.has_sift = True
        except:
            self.sift = None
            self.has_sift = False
        
        self.stats = defaultdict(int)
    
    def adaptive_preprocess(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            processed = img.copy()
            processed = cv2.bilateralFilter(processed, 5, 25, 25)
            
            if contrast < 40 or brightness < 70 or brightness > 200:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clip_limit = 2.5 if contrast < 30 else 1.8
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                processed = cv2.merge([l, a, b])
                processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
            
            if blur_score < 120:
                kernel = np.array([[0, -0.8, 0], [-0.8, 4.2, -0.8], [0, -0.8, 0]])
                sharpened = cv2.filter2D(processed, -1, kernel)
                alpha = 0.4 if blur_score < 60 else 0.25
                processed = cv2.addWeighted(processed, 1-alpha, sharpened, alpha, 0)
            
            return processed
            
        except Exception:
            return img
    
    def get_image(self, image_id, folder):
        if folder == "private_images" or folder == "query":
            base_path = self.query_images_path
        elif folder == "database_images" or folder == "database":
            base_path = self.database_images_path
        else:
            base_path = f"/mnt/e/Embedding/beit3_project/images/{folder}"
            
        path = f"{base_path}/{image_id}.jpg"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load: {path}")
        
        return self.adaptive_preprocess(img)
    
    def compute_enhanced_feature_score(self, query_img, candidate_img):
        
        gray1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        
        scores = []
        
        orb_score = self._compute_robust_orb_score(gray1, gray2)
        scores.append(orb_score)
        
        if self.has_sift:
            sift_score = self._compute_robust_sift_score(gray1, gray2)
            scores.append(sift_score * 0.85)
        
        if min(gray1.shape) > 400 and min(gray2.shape) > 400:
            small1 = cv2.resize(gray1, (gray1.shape[1]//2, gray1.shape[0]//2))
            small2 = cv2.resize(gray2, (gray2.shape[1]//2, gray2.shape[0]//2))
            small_score = self._compute_robust_orb_score(small1, small2)
            scores.append(small_score * 0.7)
        
        if self.enable_multiple_validations:
            template_score = self._compute_template_matching_score(gray1, gray2)
            scores.append(template_score * 0.4)
        
        if len(scores) >= 3:
            weights = [1.0, 0.85, 0.7, 0.4][:len(scores)]
            weighted_scores = [s * w for s, w in zip(scores, weights)]
            final_score = np.median(weighted_scores)
        else:
            final_score = np.mean(scores) if scores else 0.0
        
        return final_score
    
    def _compute_robust_orb_score(self, gray1, gray2):
        try:
            kp1, des1 = self.orb.detectAndCompute(gray1, None)
            kp2, des2 = self.orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
                return 0.0
            
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = matcher.knnMatch(des1, des2, k=2)
            
            good_matches = []
            ratio_threshold = 0.8 if len(kp1) < 300 else 0.75
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 6:
                return 0.0
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=3.5,
                confidence=0.995,
                maxIters=2000
            )
            
            if mask is None:
                return 0.0
            
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(good_matches)
            
            base_score = inliers * 0.6 + inlier_ratio * 40
            
            if homography is not None and inliers > 8:
                det = np.linalg.det(homography[:2, :2])
                if 0.25 < det < 4.0:
                    base_score += 8
                    
                    if inliers > 10:
                        inlier_pts = src_pts[mask.ravel() == 1]
                        distribution_score = self._compute_spatial_distribution_v2(inlier_pts, gray1.shape)
                        base_score += distribution_score * 12
            
            return base_score
            
        except Exception:
            return 0.0
    
    def _compute_robust_sift_score(self, gray1, gray2):
        try:
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
                return 0.0
            
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4:
                return 0.0
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            
            if mask is None:
                return 0.0
            
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(good_matches)
            
            score = inliers * 0.8 + inlier_ratio * 30
            
            return score
            
        except Exception:
            return 0.0
    
    def _compute_template_matching_score(self, gray1, gray2):
        try:
            target_size = 256
            img1_resized = cv2.resize(gray1, (target_size, target_size))
            img2_resized = cv2.resize(gray2, (target_size, target_size))
            
            methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
            scores = []
            
            for method in methods:
                result = cv2.matchTemplate(img1_resized, img2_resized, method)
                score = result[0, 0]
                scores.append(score)
            
            avg_score = np.mean(scores)
            return max(0, avg_score * 35)
            
        except:
            return 0.0
    
    def _compute_spatial_distribution_v2(self, points, img_shape):
        if len(points) < 6:
            return 0.0
        
        grid_size = 4
        h, w = img_shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        grid_counts = np.zeros((grid_size, grid_size))
        for pt in points:
            x, y = pt
            grid_x = min(int(x // cell_w), grid_size - 1)
            grid_y = min(int(y // cell_h), grid_size - 1)
            grid_counts[grid_y, grid_x] += 1
        
        cells_with_points = np.sum(grid_counts > 0)
        distribution_ratio = cells_with_points / (grid_size * grid_size)
        
        if cells_with_points > 0:
            concentration = np.std(grid_counts[grid_counts > 0])
            penalty = np.exp(-concentration / len(points))
            distribution_ratio *= penalty
        
        return distribution_ratio
    
    def compute_reranking_confidence_v3(self, candidate_scores):
        
        if len(candidate_scores) < 2:
            return 0.0, "Insufficient candidates"
        
        scores = [item['score'] for item in candidate_scores]
        
        current_top_score = scores[0]
        best_score = max(scores)
        best_idx = scores.index(best_score)
        
        if best_idx == 0:
            return 0.0, "Current top is already best"
        
        score_improvement = best_score - current_top_score
        relative_improvement = score_improvement / max(current_top_score, 1)
        
        if best_score < self.feature_quality_threshold:
            return 0.0, f"Best score too low ({best_score:.1f})"
        
        if score_improvement < 8:
            return 0.0, f"Improvement too small ({score_improvement:.1f})"
        
        confidence = 0.0
        
        if relative_improvement > 0.2:
            confidence += 0.4
        elif relative_improvement > 0.1:
            confidence += 0.2
        
        if best_score > 35:
            confidence += 0.3
        elif best_score > 25:
            confidence += 0.2
        
        if score_improvement > 15:
            confidence += 0.2
        
        if best_idx == 2:
            confidence -= 0.1
        
        if len(scores) == 3:
            score_std = np.std(scores)
            if score_std > 10:
                confidence += 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        reason = f"Improvement: {score_improvement:.1f} ({relative_improvement:.1%}), Best: {best_score:.1f}"
        
        return confidence, reason
    
    def rerank_top3_optimized(self, json_path):
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        reranked = []
        
        for query in tqdm(data, desc="Stage 3"):
            query_id = query["query_id"]
            results = query["results"]
            
            try:
                query_img = self.get_image(query_id, "private_images")
                
                top3 = results[:3]
                others = results[3:]
                
                if len(top3) < 2:
                    reranked.append(query)
                    self.stats['insufficient_candidates'] += 1
                    continue
                
                candidate_scores = []
                
                for i, candidate in enumerate(top3):
                    try:
                        candidate_img = self.get_image(candidate["image_id"], "database_images")
                        score = self.compute_enhanced_feature_score(query_img, candidate_img)
                        
                        position_bias = (3 - i) * 1.5
                        final_score = score + position_bias
                        
                        candidate_scores.append({
                            'index': i,
                            'score': score,
                            'final_score': final_score,
                            'data': candidate
                        })
                        
                    except Exception:
                        candidate_scores.append({
                            'index': i,
                            'score': 0,
                            'final_score': 0,
                            'data': candidate
                        })
                        self.stats['candidate_errors'] += 1
                
                confidence, reason = self.compute_reranking_confidence_v3(candidate_scores)
                
                if confidence >= self.confidence_threshold:
                    candidate_scores.sort(key=lambda x: x['final_score'], reverse=True)
                    self.stats['reranked'] += 1
                    
                    new_order = [item['index'] for item in candidate_scores]
                    if new_order[0] != 0:
                        self.stats['top1_changed'] += 1
                        
                        if new_order[0] == 1:
                            self.stats['pos2_to_pos1'] += 1
                        elif new_order[0] == 2:
                            self.stats['pos3_to_pos1'] += 1
                else:
                    self.stats['kept_original'] += 1
                
                final_top3 = [item['data'] for item in candidate_scores]
                
            except Exception:
                final_top3 = top3
                self.stats['query_errors'] += 1
            
            reranked.append({
                "query_id": query_id,
                "results": final_top3 + others
            })
        
        return reranked

class Top1VsTop4Reranker:
    
    def __init__(self, 
                 confidence_threshold=0.25,
                 feature_quality_threshold=18,
                 enable_multiple_validations=True,
                 query_images_path=None,
                 database_images_path=None):
        
        self.confidence_threshold = confidence_threshold
        self.feature_quality_threshold = feature_quality_threshold
        self.enable_multiple_validations = enable_multiple_validations
        
        self.query_images_path = query_images_path or "/mnt/e/Embedding/beit3_project/images/private_images"
        self.database_images_path = database_images_path or "/mnt/e/Embedding/beit3_project/images/database_images"
        
        self.orb = cv2.ORB_create(
            nfeatures=2500,
            scaleFactor=1.15,
            nlevels=10,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        
        try:
            self.sift = cv2.SIFT_create(
                nfeatures=1200,
                nOctaveLayers=4,
                contrastThreshold=0.03,
                edgeThreshold=8,
                sigma=1.6
            )
            self.has_sift = True
        except:
            self.sift = None
            self.has_sift = False
        
        self.stats = defaultdict(int)
    
    def adaptive_preprocess(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            processed = img.copy()
            processed = cv2.bilateralFilter(processed, 5, 25, 25)
            
            if contrast < 40 or brightness < 70 or brightness > 200:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clip_limit = 2.5 if contrast < 30 else 1.8
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                processed = cv2.merge([l, a, b])
                processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
            
            if blur_score < 120:
                kernel = np.array([[0, -0.8, 0], [-0.8, 4.2, -0.8], [0, -0.8, 0]])
                sharpened = cv2.filter2D(processed, -1, kernel)
                alpha = 0.4 if blur_score < 60 else 0.25
                processed = cv2.addWeighted(processed, 1-alpha, sharpened, alpha, 0)
            
            return processed
            
        except Exception:
            return img
    
    def get_image(self, image_id, folder):
        if folder == "private_images" or folder == "query":
            base_path = self.query_images_path
        elif folder == "database_images" or folder == "database":
            base_path = self.database_images_path
        else:
            base_path = f"/mnt/e/Embedding/beit3_project/images/{folder}"
            
        path = f"{base_path}/{image_id}.jpg"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load: {path}")
        
        return self.adaptive_preprocess(img)
    
    def compute_enhanced_feature_score(self, query_img, candidate_img):
        
        gray1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        
        scores = []
        
        orb_score = self._compute_robust_orb_score(gray1, gray2)
        scores.append(orb_score)
        
        if self.has_sift:
            sift_score = self._compute_robust_sift_score(gray1, gray2)
            scores.append(sift_score * 0.85)
        
        if min(gray1.shape) > 400 and min(gray2.shape) > 400:
            small1 = cv2.resize(gray1, (gray1.shape[1]//2, gray1.shape[0]//2))
            small2 = cv2.resize(gray2, (gray2.shape[1]//2, gray2.shape[0]//2))
            small_score = self._compute_robust_orb_score(small1, small2)
            scores.append(small_score * 0.7)
        
        if self.enable_multiple_validations:
            template_score = self._compute_template_matching_score(gray1, gray2)
            scores.append(template_score * 0.4)
        
        if len(scores) >= 3:
            weights = [1.0, 0.85, 0.7, 0.4][:len(scores)]
            weighted_scores = [s * w for s, w in zip(scores, weights)]
            final_score = np.median(weighted_scores)
        else:
            final_score = np.mean(scores) if scores else 0.0
        
        return final_score
    
    def _compute_robust_orb_score(self, gray1, gray2):
        try:
            kp1, des1 = self.orb.detectAndCompute(gray1, None)
            kp2, des2 = self.orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
                return 0.0
            
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = matcher.knnMatch(des1, des2, k=2)
            
            good_matches = []
            ratio_threshold = 0.8 if len(kp1) < 300 else 0.75
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 6:
                return 0.0
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=3.5,
                confidence=0.995,
                maxIters=2000
            )
            
            if mask is None:
                return 0.0
            
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(good_matches)
            
            base_score = inliers * 0.6 + inlier_ratio * 40
            
            if homography is not None and inliers > 8:
                det = np.linalg.det(homography[:2, :2])
                if 0.25 < det < 4.0:
                    base_score += 8
                    
                    if inliers > 10:
                        inlier_pts = src_pts[mask.ravel() == 1]
                        distribution_score = self._compute_spatial_distribution_v4(inlier_pts, gray1.shape)
                        base_score += distribution_score * 12
            
            return base_score
            
        except Exception:
            return 0.0
    
    def _compute_robust_sift_score(self, gray1, gray2):
        try:
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
                return 0.0
            
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4:
                return 0.0
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            
            if mask is None:
                return 0.0
            
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(good_matches)
            
            score = inliers * 0.8 + inlier_ratio * 30
            
            return score
            
        except Exception:
            return 0.0
    
    def _compute_template_matching_score(self, gray1, gray2):
        try:
            target_size = 256
            img1_resized = cv2.resize(gray1, (target_size, target_size))
            img2_resized = cv2.resize(gray2, (target_size, target_size))
            
            methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
            scores = []
            
            for method in methods:
                result = cv2.matchTemplate(img1_resized, img2_resized, method)
                score = result[0, 0]
                scores.append(score)
            
            avg_score = np.mean(scores)
            return max(0, avg_score * 35)
            
        except:
            return 0.0
    
    def _compute_spatial_distribution_v4(self, points, img_shape):
        if len(points) < 6:
            return 0.0
        
        grid_size = 4
        h, w = img_shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        grid_counts = np.zeros((grid_size, grid_size))
        for pt in points:
            x, y = pt
            grid_x = min(int(x // cell_w), grid_size - 1)
            grid_y = min(int(y // cell_h), grid_size - 1)
            grid_counts[grid_y, grid_x] += 1
        
        cells_with_points = np.sum(grid_counts > 0)
        distribution_ratio = cells_with_points / (grid_size * grid_size)
        
        if cells_with_points > 0:
            concentration = np.std(grid_counts[grid_counts > 0])
            penalty = np.exp(-concentration / len(points))
            distribution_ratio *= penalty
        
        return distribution_ratio
    
    def compute_reranking_confidence_v4(self, top1_score, top4_score):
        
        score_improvement = top4_score - top1_score
        
        if top1_score <= 0:
            relative_improvement = float('inf') if top4_score > 0 else 0
        else:
            relative_improvement = score_improvement / top1_score
        
        if top4_score < self.feature_quality_threshold:
            return 0.0, f"Score too low"
        
        if score_improvement < 8:
            return 0.0, f"Improvement too small"
        
        confidence = 0.0
        
        if relative_improvement > 0.3:
            confidence += 0.5
        elif relative_improvement > 0.2:
            confidence += 0.4
        elif relative_improvement > 0.1:
            confidence += 0.2
        
        if top4_score > 35:
            confidence += 0.3
        elif top4_score > 25:
            confidence += 0.2
        
        if score_improvement > 20:
            confidence += 0.2
        elif score_improvement > 15:
            confidence += 0.1
        
        confidence -= 0.15
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, f"Improvement: {score_improvement:.1f}"
    
    def rerank_top1_vs_top4(self, json_path):
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        reranked = []
        
        for query in tqdm(data, desc="Stage 4"):
            query_id = query["query_id"]
            results = query["results"]
            
            try:
                query_img = self.get_image(query_id, "private_images")
                
                if len(results) < 4:
                    reranked.append(query)
                    self.stats['insufficient_candidates'] += 1
                    continue
                
                top1_candidate = results[0]
                top4_candidate = results[3]
                
                try:
                    top1_img = self.get_image(top1_candidate["image_id"], "database_images")
                    top1_score = self.compute_enhanced_feature_score(query_img, top1_img)
                except Exception:
                    top1_score = 0
                    self.stats['top1_errors'] += 1
                
                try:
                    top4_img = self.get_image(top4_candidate["image_id"], "database_images")
                    top4_score = self.compute_enhanced_feature_score(query_img, top4_img)
                except Exception:
                    top4_score = 0
                    self.stats['top4_errors'] += 1
                
                confidence, reason = self.compute_reranking_confidence_v4(top1_score, top4_score)
                
                if confidence >= self.confidence_threshold:
                    new_results = [top4_candidate] + results[1:3] + [top1_candidate] + results[4:]
                    self.stats['promoted_top4'] += 1
                else:
                    new_results = results
                    self.stats['kept_original'] += 1
                
            except Exception:
                new_results = results
                self.stats['query_errors'] += 1
            
            reranked.append({
                "query_id": query_id,
                "results": new_results
            })
        
        return reranked

def stage2_enhanced_reranking(input_file, output_file, query_images_path=None, database_images_path=None):
    print("Stage 2: Enhanced reranking")
    
    reranker = EnhancedReranker(query_images_path, database_images_path)
    results = reranker.rerank_top2(input_file)
    
    save_results(results, output_file)
    print(f"Saved to {output_file}")
    return results 

def stage3_optimized_top3(input_file, output_file, query_images_path=None, database_images_path=None):
    print("Stage 3: Top-3 optimization")
    
    reranker = OptimizedTop3Reranker(
        confidence_threshold=0.25,
        feature_quality_threshold=18,
        enable_multiple_validations=True,
        query_images_path=query_images_path,
        database_images_path=database_images_path
    )
    
    results = reranker.rerank_top3_optimized(input_file)
    
    save_results(results, output_file)
    print(f"Saved to {output_file}")
    return results 

def stage4_top1_vs_top4(input_file, output_file, query_images_path=None, database_images_path=None):
    print("Stage 4: Top1 vs Top4")
    
    reranker = Top1VsTop4Reranker(
        confidence_threshold=0.25,
        feature_quality_threshold=18,
        enable_multiple_validations=True,
        query_images_path=query_images_path,
        database_images_path=database_images_path
    )
    
    results = reranker.rerank_top1_vs_top4(input_file)
    
    save_results(results, output_file)
    print(f"Saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Re-ranking Pipeline')
    
    parser.add_argument('--model1', type=str, default='retrieval_results/beit3_coco_base.json')
    parser.add_argument('--model2', type=str, default='retrieval_results/siglipso.json')
    parser.add_argument('--model3', type=str, default='retrieval_results/beit3_base_ft.json')
    parser.add_argument('--model_names', type=str, nargs='+', default=None)
    parser.add_argument('--weights', type=float, nargs='+', default=None)
    
    parser.add_argument('--query_images_path', type=str, 
                       default='/mnt/e/Embedding/beit3_project/images/private_images')
    parser.add_argument('--database_images_path', type=str, 
                       default='/mnt/e/Embedding/beit3_project/images/database_images')
    
    parser.add_argument('--stage1_output', type=str, default='reranked_stage1.json')
    parser.add_argument('--stage2_output', type=str, default='reranked_stage2.json')
    parser.add_argument('--stage3_output', type=str, default='reranked_stage3.json')
    parser.add_argument('--final_output', type=str, default='reranked_final.json')
    
    parser.add_argument('--skip_stage1', action='store_true')
    parser.add_argument('--skip_stage2', action='store_true')
    parser.add_argument('--skip_stage3', action='store_true')
    parser.add_argument('--only_stage', type=int, choices=[1, 2, 3, 4])
    
    args = parser.parse_args()
    
    try:
        print("Re-ranking pipeline started")
        
        if args.only_stage is None or args.only_stage == 1:
            if not args.skip_stage1:
                stage1_results = stage1_ensemble(
                    args.model1, args.model2, args.model3, 
                    args.stage1_output, args.weights, args.model_names
                )
                if stage1_results is None:
                    return
        
        if args.only_stage is None or args.only_stage == 2:
            if not args.skip_stage2:
                stage2_enhanced_reranking(args.stage1_output, args.stage2_output, 
                                        args.query_images_path, args.database_images_path)
        
        if args.only_stage is None or args.only_stage == 3:
            if not args.skip_stage3:
                stage3_optimized_top3(args.stage2_output, args.stage3_output,
                                    args.query_images_path, args.database_images_path)
        
        if args.only_stage is None or args.only_stage == 4:
            stage4_top1_vs_top4(args.stage3_output, args.final_output,
                              args.query_images_path, args.database_images_path)
        
        if args.only_stage is None:
            print(f"Pipeline completed! Final result: {args.final_output}")
        else:
            print(f"Stage {args.only_stage} completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()