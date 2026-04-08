import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Исходные папки (5 классов)
TRAIN_5 = os.path.join(PROCESSED_DIR, "train")
VAL_5 = os.path.join(PROCESSED_DIR, "val")

# Новые папки для бинарной классификации
BINARY_DIR = os.path.join(BASE_DIR, "data", "binary")
TRAIN_BIN = os.path.join(BINARY_DIR, "train")
VAL_BIN = os.path.join(BINARY_DIR, "val")

# Маппинг: старый класс -> новый класс
# caries и deep_caries -> caries
# healthy, impacted, periapical_lesion -> healthy
MAPPING = {
    "caries": "caries",
    "deep_caries": "caries",
    "healthy": "healthy",
    "impacted": "healthy",
    "periapical_lesion": "healthy"
}

def copy_mapped(src_dir, dst_dir):
    """Копирует файлы из src_dir в dst_dir, объединяя классы согласно MAPPING"""
    for old_class, new_class in MAPPING.items():
        src_class_path = os.path.join(src_dir, old_class)
        if not os.path.exists(src_class_path):
            print(f"Предупреждение: папка {src_class_path} не найдена")
            continue
        dst_class_path = os.path.join(dst_dir, new_class)
        os.makedirs(dst_class_path, exist_ok=True)
        for fname in os.listdir(src_class_path):
            src = os.path.join(src_class_path, fname)
            dst = os.path.join(dst_class_path, fname)
            shutil.copy2(src, dst)
        print(f"Скопировано {len(os.listdir(src_class_path))} файлов из {old_class} -> {new_class}")

# Копируем train и val
print("Обработка train...")
copy_mapped(TRAIN_5, TRAIN_BIN)
print("Обработка val...")
copy_mapped(VAL_5, VAL_BIN)

# Статистика
print("\n--- Статистика после объединения ---")
for split, split_path in [("Train", TRAIN_BIN), ("Val", VAL_BIN)]:
    print(f"\n{split}:")
    for class_name in ["caries", "healthy"]:
        path = os.path.join(split_path, class_name)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {class_name}: {count}")