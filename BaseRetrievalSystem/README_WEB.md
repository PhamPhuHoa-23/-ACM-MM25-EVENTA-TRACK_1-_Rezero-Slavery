# Image Retrieval Visualization Web App

Web interface để visualize kết quả từ hệ thống retrieval với giao diện đẹp và dễ sử dụng.

## Cài đặt

```bash
pip install -r requirements_web.txt
```

## Chạy Web App

### Lệnh cơ bản:
```bash
python app.py --database_dir ./data/database --query_dir ./data/query
```

### Lệnh đầy đủ:
```bash
python app.py \
  --database_dir ./data/database \
  --query_dir ./data/query \
  --results_file retrieval_results.json \
  --article_url_file database_article_to_url.json \
  --image_article_file database_images_to_article_v.0.1.json \
  --host 127.0.0.1 \
  --port 5000 \
  --debug
```

## Tham số

- `--database_dir`: Thư mục chứa ảnh database (default: `./data/database`)
- `--query_dir`: Thư mục chứa ảnh query (default: `./data/query`)
- `--results_file`: File JSON kết quả retrieval (default: `retrieval_results.json`)
- `--article_url_file`: File JSON mapping article ID → URL (default: `database_article_to_url.json`)
- `--image_article_file`: File JSON mapping image ID → article ID (default: `database_images_to_article_v.0.1.json`)
- `--host`: Host để chạy web server (default: `127.0.0.1`)
- `--port`: Port để chạy web server (default: `5000`)
- `--debug`: Chạy ở chế độ debug

## Giao diện

### Query Navigation
- **Previous/Current/Next**: Ảnh query hiện tại ở giữa (lớn), previous/next ở hai bên (nhỏ, mờ)
- **Navigation**: Nút Previous/Next hoặc phím mũi tên ←/→
- **Jump**: Nhập số thứ tự query để nhảy trực tiếp

### Results Display
- **Grid 2x5**: Hiển thị 10 kết quả retrieval trong lưới 2 hàng × 5 cột
- **Score**: Hiển thị điểm similarity trên mỗi ảnh
- **Article Info**: Hiển thị Article ID và link đến bài báo gốc
- **Ranking**: Thứ hạng (#1 → #10)

## Features

- ✅ Responsive design với Tailwind CSS
- ✅ Navigation bằng keyboard (←/→)
- ✅ Hover effects và animations
- ✅ Placeholder images khi không tìm thấy file
- ✅ External links đến articles
- ✅ Score hiển thị với 4 chữ số thập phân
- ✅ Query ID và image ID tooltips

## Cấu trúc Files

```
├── app.py                              # Flask web application
├── templates/
│   └── index.html                      # Main HTML template
├── requirements_web.txt                # Python dependencies
├── setup_images.py                     # Utility to copy images
├── retrieval_results.json              # Results from retrieval system
├── database_article_to_url.json        # Article ID → URL mapping
└── database_images_to_article_v.0.1.json # Image ID → Article ID mapping
```

## Troubleshooting

### Không hiển thị ảnh?
1. Kiểm tra đường dẫn `--database_dir` và `--query_dir`
2. Đảm bảo ảnh có định dạng `.jpg`, `.jpeg`, hoặc `.png`
3. Check console browser để xem lỗi 404

### Không có kết quả?
1. Đảm bảo `retrieval_results.json` tồn tại
2. Chạy retrieval system trước để tạo results
3. Check format của JSON file

### Links không hoạt động?
1. Kiểm tra file `database_article_to_url.json`
2. Đảm bảo article IDs khớp nhau

## Ví dụ chạy hoàn chỉnh

```bash
# 1. Chạy retrieval system để tạo kết quả
python retrieval_system.py --query_dir ./data/query --vectors_dir ./data/database

# 2. Chạy web visualization
python app.py --database_dir ./data/database --query_dir ./data/query

# 3. Mở browser: http://127.0.0.1:5000
``` 