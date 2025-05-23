import os
import shutil
import argparse
from pathlib import Path

def copy_images(source_dir, dest_dir, extensions=['.jpg', '.jpeg', '.png']):
    """Copy images from source to destination directory"""
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return 0
    
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = 0
    for file in os.listdir(source_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(dest_dir, file)
            
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
    
    print(f"Copied {copied} images to {dest_dir}")
    return copied

def main():
    parser = argparse.ArgumentParser(description='Setup images for web visualization')
    parser.add_argument('--database_dir', type=str, required=True, help='Source database images directory')
    parser.add_argument('--query_dir', type=str, required=True, help='Source query images directory')
    parser.add_argument('--static_database', type=str, default='static/database', help='Destination database static directory')
    parser.add_argument('--static_query', type=str, default='static/query', help='Destination query static directory')
    
    args = parser.parse_args()
    
    print("Setting up images for web visualization...")
    
    # Copy database images
    db_copied = copy_images(args.database_dir, args.static_database)
    
    # Copy query images
    query_copied = copy_images(args.query_dir, args.static_query)
    
    print(f"\nSetup complete!")
    print(f"- Database images: {db_copied}")
    print(f"- Query images: {query_copied}")
    
    if db_copied == 0 and query_copied == 0:
        print("\nNote: No images were copied. You can run the web app directly if image directories are accessible.")

if __name__ == '__main__':
    main() 