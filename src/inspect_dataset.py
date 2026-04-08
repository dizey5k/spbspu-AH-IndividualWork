import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "dental_dataset")

IMAGES_CUT = os.path.join(RAW_DATASET_PATH, "images_cut")
BBOXES_CARIES = os.path.join(RAW_DATASET_PATH, "annotations", "bboxes_caries")
BBOXES_TEETH = os.path.join(RAW_DATASET_PATH, "annotations", "bboxes_teeth")

# Все изображения
all_images = [f for f in os.listdir(IMAGES_CUT) if f.endswith('.png')]
print(f"Всего изображений в images_cut: {len(all_images)}")

# Какие имеют файлы в caries
caries_files = set(os.path.splitext(f)[0] for f in os.listdir(BBOXES_CARIES) if f.endswith('.txt'))
# Какие имеют файлы в teeth
teeth_files = set(os.path.splitext(f)[0] for f in os.listdir(BBOXES_TEETH) if f.endswith('.txt'))

# Для каждого изображения проверим
healthy_count = 0
caries_only_count = 0
teeth_only_count = 0
both_count = 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "dental_dataset")

IMAGES_FULL = os.path.join(RAW_DATASET_PATH, "images")
if os.path.exists(IMAGES_FULL):
    images_full_list = [f for f in os.listdir(IMAGES_FULL) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Папка 'images' найдена. Количество изображений: {len(images_full_list)}")
else:
    print("Папка 'images' не найдена")

for img in all_images:
    name = os.path.splitext(img)[0]
    has_caries = name in caries_files
    has_teeth = name in teeth_files
    
    if has_caries and has_teeth:
        both_count += 1
    elif has_caries and not has_teeth:
        caries_only_count += 1
    elif not has_caries and has_teeth:
        teeth_only_count += 1
    else:
        healthy_count += 1
        print(f"Возможно здоровый: {img} (нет ни caries, ни teeth аннотаций)")

print(f"\n--- Статистика ---")
print(f"Есть и caries, и teeth: {both_count}")
print(f"Только caries: {caries_only_count}")
print(f"Только teeth: {teeth_only_count}")
print(f"Нет аннотаций (возможно здоровые): {healthy_count}")

# Дополнительно: проверим содержимое нескольких файлов caries
print("\nПример содержимого bboxes_caries (первые 3 файла):")
for i, f in enumerate(list(os.listdir(BBOXES_CARIES))[:3]):
    path = os.path.join(BBOXES_CARIES, f)
    with open(path, 'r') as file:
        content = file.read().strip()
        print(f"{f}: {content[:100]}...")