import os
from PIL import Image

def find_broken_images(folder):
    broken = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.convert("RGB")  # вот здесь может упасть
                except Exception as e:
                    print(f"BROKEN: {path} - {e}")
                    broken.append(path)
    return broken

broken_train = find_broken_images("data/binary/train")
broken_val = find_broken_images("data/binary/val")
print(f"Total broken in train: {len(broken_train)}, val: {len(broken_val)}")