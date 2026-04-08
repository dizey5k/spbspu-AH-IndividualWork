import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import csv
from datetime import datetime

# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень проекта
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "dental_binary_model.pth")
CLASS_NAMES = ['caries', 'healthy']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформации (как при валидации)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------
# Загрузка модели
# ----------------------------------------------------------------------
def load_model(model_path):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ----------------------------------------------------------------------
# Предсказание одного изображения
# ----------------------------------------------------------------------
def predict_image(model, image_path):
    """Возвращает (pred_class, confidence, probs_list)"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence, probabilities.cpu().numpy()[0]

# ----------------------------------------------------------------------
# Сканирование папки на изображения
# ----------------------------------------------------------------------
def get_image_files(folder, recursive=False):
    image_files = []
    if recursive:
        for root, _, files in os.walk(folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if os.path.isfile(full_path) and any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_files.append(full_path)
    return sorted(image_files)

# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Predict dental caries for all images in a folder')
    parser.add_argument('folder', type=str, help='Path to folder containing images')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to model weights')
    parser.add_argument('--recursive', action='store_true', help='Scan subfolders recursively')
    parser.add_argument('--show_probs', action='store_true', help='Show probabilities for each class')
    parser.add_argument('--csv', type=str, help='Save results to CSV file (e.g., results.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory.")
        return

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print(f"Model loaded. Device: {device}")
    print(f"Scanning folder: {args.folder}")
    print(f"Recursive mode: {args.recursive}\n")

    image_paths = get_image_files(args.folder, args.recursive)
    if not image_paths:
        print("No image files found.")
        return

    print(f"Found {len(image_paths)} image(s)\n")
    results = []

    for i, img_path in enumerate(image_paths, 1):
        try:
            pred_class, confidence, probs = predict_image(model, img_path)
            rel_path = os.path.relpath(img_path, args.folder)
            results.append({
                'image': rel_path,
                'prediction': pred_class,
                'confidence': f"{confidence:.2%}",
                'confidence_value': confidence,
                'prob_caries': probs[0],
                'prob_healthy': probs[1]
            })
            print(f"[{i}/{len(image_paths)}] {rel_path}")
            print(f"   -> {pred_class.upper()} (confidence: {confidence:.2%})")
            if args.show_probs:
                print(f"      caries: {probs[0]:.2%}, healthy: {probs[1]:.2%}")
        except Exception as e:
            print(f"[{i}/{len(image_paths)}] {img_path} - ERROR: {e}")

    # Сохраняем CSV, если нужно
    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'prediction', 'confidence', 'prob_caries', 'prob_healthy'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'image': r['image'],
                    'prediction': r['prediction'],
                    'confidence': r['confidence'],
                    'prob_caries': f"{r['prob_caries']:.4f}",
                    'prob_healthy': f"{r['prob_healthy']:.4f}"
                })
        print(f"\nResults saved to {args.csv}")

    # Краткая статистика
    pred_counts = {cls: sum(1 for r in results if r['prediction'] == cls) for cls in CLASS_NAMES}
    print("\n--- Summary ---")
    for cls, cnt in pred_counts.items():
        print(f"{cls}: {cnt} images ({cnt/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()