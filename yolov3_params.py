# save_as: download_yolo.py
import urllib.request
import os

print("Скачиваю файлы YOLOv3...")

# Создаем папку
os.makedirs("data", exist_ok=True)

# 1. Конфигурационный файл
print("1. yolov3.cfg...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "data/yolov3.cfg"
)

# 2. Имена классов COCO (80 классов)
print("2. coco.names...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "data/coco.names"
)

print("✓ Файлы 1 и 2 скачаны")

print("\n3. ВАЖНО: файл yolov3.weights (246 МБ) нужно скачать ВРУЧНУЮ:")
print("   Перейдите по ссылке: https://pjreddie.com/media/files/yolov3.weights")
print("   Сохраните файл в папку: data/yolov3.weights")
print("\nПосле скачивания запустите: python main.py")