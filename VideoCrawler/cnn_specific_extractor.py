import time
import json
import re
import os
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def setup_driver():
    """Set up an enhanced Selenium WebDriver specifically for CNN."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--mute-audio")
    
    # More sophisticated user agent
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Enable network logging and JavaScript console logs
    chrome_options.set_capability("goog:loggingPrefs", {
        "performance": "ALL",
        "browser": "ALL"
    })
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set page load timeout
    driver.set_page_load_timeout(30)
    
    return driver

def extract_video_url_from_cnn(driver, url):
    """Extract video URLs using CNN-specific techniques."""
    video_urls = []
    
    # 1. First try: Look for specific CNN player structures in the DOM
    try:
        # Wait for CNN video player to load (max 10 seconds)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".video-player-container, .media__video, .media__video-player, .videoPlayer"))
            )
            print("CNN video player element found")
        except TimeoutException:
            print("Timeout waiting for video player element")
        
        # Try to play the video by clicking play button
        play_buttons = driver.find_elements(By.CSS_SELECTOR, 
            ".video-js .vjs-big-play-button, .player-overlay-play-button, .js-video-play-button, [data-analytics='videoPlay']"
        )
        if play_buttons:
            for button in play_buttons[:1]:  # Only click the first button found
                try:
                    driver.execute_script("arguments[0].scrollIntoView(true);", button)
                    button.click()
                    print("Clicked play button")
                    time.sleep(5)  # Wait for video to start playing
                    break
                except Exception as e:
                    print(f"Failed to click play button: {str(e)}")
    except Exception as e:
        print(f"Error during video player interaction: {str(e)}")
    
    # 2. Extract from network logs
    try:
        print("Analyzing network logs for video content...")
        logs = driver.get_log("performance")
        video_extensions = [".mp4", ".m3u8", ".ts", ".webm", ".mpd"]
        video_domains = ["cdn.cnn.com", "cnnios-f.akamaihd.net", "cnn-f.akamaihd.net", "cnnlive.warnermediacdn.com"]
        
        for log in logs:
            try:
                log_entry = json.loads(log["message"])
                if "params" in log_entry and "request" in log_entry["params"]:
                    request = log_entry["params"]["request"]
                    if "url" in request:
                        request_url = request["url"]
                        # Check if URL is likely a video
                        if (any(ext in request_url.lower() for ext in video_extensions) or
                            any(domain in request_url.lower() for domain in video_domains)):
                            if request_url not in video_urls:
                                video_urls.append(request_url)
                                print(f"Found video URL in network log: {request_url}")
            except:
                continue
    except Exception as e:
        print(f"Error extracting from network logs: {str(e)}")
    
    # 3. Look for CNN video metadata in page source
    try:
        print("Searching for CNN video metadata in page source...")
        page_source = driver.page_source
        
        # Method 1: Look for metadata in videoPlayer configuration
        config_pattern = r'"videoPlayer":\s*{[^}]*"url"\s*:\s*"([^"]+)"'
        config_matches = re.findall(config_pattern, page_source)
        for url in config_matches:
            # Clean up URL (remove escaped characters)
            clean_url = url.replace('\\', '')
            if clean_url not in video_urls:
                video_urls.append(clean_url)
                print(f"Found video URL in player config: {clean_url}")
        
        # Method 2: Look for specific patterns in CNN's JavaScript
        script_elements = driver.find_elements(By.TAG_NAME, "script")
        for script in script_elements:
            try:
                script_content = script.get_attribute("innerHTML")
                
                # Look for M3U8 playlist URLs (common format for CNN videos)
                m3u8_pattern = r'https?://[^\s"\']+\.m3u8[^\s"\']*'
                m3u8_urls = re.findall(m3u8_pattern, script_content)
                for url in m3u8_urls:
                    if url not in video_urls:
                        video_urls.append(url)
                        print(f"Found M3U8 URL in script: {url}")
                
                # Look for MP4 video URLs
                mp4_pattern = r'https?://[^\s"\']+\.mp4[^\s"\']*'
                mp4_urls = re.findall(mp4_pattern, script_content)
                for url in mp4_urls:
                    if url not in video_urls:
                        video_urls.append(url)
                        print(f"Found MP4 URL in script: {url}")
                        
                # Check for CNN video data structures
                if "CNN.VideoPlayer.metadata" in script_content:
                    print("Found CNN.VideoPlayer.metadata, extracting...")
                    metadata_pattern = r'"contentUrl"\s*:\s*"([^"]+)"'
                    metadata_matches = re.findall(metadata_pattern, script_content)
                    for url in metadata_matches:
                        clean_url = url.replace('\\', '')
                        if clean_url not in video_urls:
                            video_urls.append(clean_url)
                            print(f"Found video URL in CNN metadata: {clean_url}")
            except:
                continue
    except Exception as e:
        print(f"Error extracting from page scripts: {str(e)}")
    
    # 4. Check for source elements explicitly for CNN
    try:
        # Look for video and source tags with more specific selectors for CNN
        video_elements = driver.find_elements(By.CSS_SELECTOR, 
            "video, .media__video video, .media__video-player video, .videoPlayer video"
        )
        
        for video in video_elements:
            try:
                # Get src attribute
                src = video.get_attribute("src")
                if src and not src.startswith("blob:") and src not in video_urls:
                    video_urls.append(src)
                    print(f"Found direct video source: {src}")
                
                # Get source elements
                source_elements = video.find_elements(By.TAG_NAME, "source")
                for source in source_elements:
                    src = source.get_attribute("src")
                    if src and not src.startswith("blob:") and src not in video_urls:
                        video_urls.append(src)
                        print(f"Found video source from source tag: {src}")
                        
                # Check for data-src attribute (sometimes used by CNN)
                data_src = video.get_attribute("data-src")
                if data_src and data_src not in video_urls:
                    video_urls.append(data_src)
                    print(f"Found video data-src attribute: {data_src}")
            except:
                continue
    except Exception as e:
        print(f"Error checking video elements: {str(e)}")
    
    # 5. Look for video in browser console logs
    try:
        console_logs = driver.get_log("browser")
        for log in console_logs:
            # Look for URLs in console logs
            url_pattern = r'https?://[^\s"\']+\.(mp4|m3u8|ts)[^\s"\']*'
            urls = re.findall(url_pattern, log["message"])
            for url in urls:
                if url not in video_urls:
                    video_urls.append(url)
                    print(f"Found video URL in console log: {url}")
    except:
        print("Could not access console logs")
    
    # 6. Last resort: Check for iframe embedded videos (like YouTube or other players)
    try:
        iframe_elements = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframe_elements:
            iframe_src = iframe.get_attribute("src")
            if iframe_src and ("youtube.com" in iframe_src or "player" in iframe_src or "video" in iframe_src):
                if iframe_src not in video_urls:
                    video_urls.append(iframe_src)
                    print(f"Found embedded video iframe: {iframe_src}")
    except:
        pass
    
    return video_urls

def get_cnn_video(url):
    """Main function to extract CNN video URL from a page."""
    driver = None
    try:
        print(f"Processing URL: {url}")
        driver = setup_driver()
        
        # Navigate to the page
        print(f"Loading page: {url}")
        driver.get(url)
        
        # Wait for page to load and settle
        time.sleep(10)
        
        # Extract video URLs
        video_urls = extract_video_url_from_cnn(driver, url)
        
        if video_urls:
            # Filter URLs to prioritize actual video files over player URLs
            # Preference: .mp4 > .m3u8 > other formats
            mp4_urls = [url for url in video_urls if '.mp4' in url.lower()]
            m3u8_urls = [url for url in video_urls if '.m3u8' in url.lower()]
            
            if mp4_urls:
                print(f"Found {len(mp4_urls)} MP4 video URLs")
                return True, mp4_urls[0]
            elif m3u8_urls:
                print(f"Found {len(m3u8_urls)} M3U8 video URLs")
                return True, m3u8_urls[0]
            else:
                print(f"Found {len(video_urls)} other video URLs")
                return True, video_urls[0]
        else:
            print("No video URLs found")
            return False, None
            
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return False, None
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    # Test with a known CNN video page
    test_url = "https://edition.cnn.com/2021/06/15/asia/swarm-robots-hong-kong-warehouse-hnk-spc-intl/index.html"
    has_video, video_url = get_cnn_video(test_url)
    
    if has_video:
        print(f"\nSUCCESS: Found actual CNN video URL:")
        print(f"  {video_url}")
        
        # Save the result
        result = {
            "url": test_url,
            "has_video": has_video,
            "video_url": video_url
        }
        
        with open("cnn_video_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("Result saved to cnn_video_result.json")
    else:
        print("\nFAILED: Could not find any CNN video URL.") 