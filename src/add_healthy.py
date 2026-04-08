import os
import shutil
import random
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Папка с новыми здоровыми снимками
EXTRA_HEALTHY_DIR = os.path.join(BASE_DIR, "data", "extra_healthy")

# Целевые папки (бинарная структура)
BINARY_TRAIN = os.path.join(BASE_DIR, "data", "binary", "train", "healthy")
BINARY_VAL = os.path.join(BASE_DIR, "data", "binary", "val", "healthy")

# Проверяем, есть ли файлы
if not os.path.exists(EXTRA_HEALTHY_DIR):
    print(f"Ошибка: папка {EXTRA_HEALTHY_DIR} не найдена")
    exit(1)

# Получаем список файлов
healthy_files = [f for f in os.listdir(EXTRA_HEALTHY_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Найдено новых здоровых снимков: {len(healthy_files)}")

if len(healthy_files) == 0:
    print("Нет файлов для добавления")
    exit(0)

# Разделяем на train (80%) и val (20%)
train_files, val_files = train_test_split(healthy_files, test_size=0.2, random_state=42)

print(f"В train попадет: {len(train_files)}")
print(f"В val попадет: {len(val_files)}")

# Копируем в train
os.makedirs(BINARY_TRAIN, exist_ok=True)
for fname in train_files:
    src = os.path.join(EXTRA_HEALTHY_DIR, fname)
    dst = os.path.join(BINARY_TRAIN, fname)
    shutil.copy2(src, dst)

# Копируем в val
os.makedirs(BINARY_VAL, exist_ok=True)
for fname in val_files:
    src = os.path.join(EXTRA_HEALTHY_DIR, fname)
    dst = os.path.join(BINARY_VAL, fname)
    shutil.copy2(src, dst)

print("\n--- Готово! ---")
print(f"Train healthy теперь: {len(os.listdir(BINARY_TRAIN))}")
print(f"Val healthy теперь: {len(os.listdir(BINARY_VAL))}")