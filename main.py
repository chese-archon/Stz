# main_final_fixed.py - Исправлено масштабирование
import cv2
import numpy as np
import os
import sys
import time
from collections import defaultdict, deque

class YOLOv3Detector:
    def __init__(self, conf_thresh=0.5):
        self.weights = "data/yolov3.weights"
        self.config = "data/yolov3.cfg"
        self.names = "data/coco.names"
        
        if not os.path.exists(self.weights):
            print("❌ Скачайте yolov3.weights в папку 'data/'")
            print("Ссылка: https://pjreddie.com/media/files/yolov3.weights")
            sys.exit(1)
        
        with open(self.names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print("Загрузка YOLOv3...")
        self.net = cv2.dnn.readNet(self.weights, self.config)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.conf_thresh = conf_thresh
        print(f"✓ YOLOv3 загружен ({len(self.classes)} классов)")
    
    def detect(self, frame, original_size):
        """
        Детектирует объекты на кадре
        original_size: (ширина, высота) оригинального кадра
        """
        height, width = frame.shape[:2]
        orig_width, orig_height = original_size
        
        # Запоминаем масштаб
        scale_x = orig_width / width
        scale_y = orig_height / height
        
        # Подготовка для YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), 
            (0, 0, 0), swapRB=True, crop=False
        )
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_thresh:
                    # Координаты относительно уменьшенного изображения
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, 0.4)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # МАСШТАБИРУЕМ координаты к оригинальному размеру!
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]]
                })
        
        return detections

class SimpleTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        
    def update(self, detections):
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        results = []
        used_detections = set()
        
        # Сначала ищем соответствия для существующих треков
        for track_id, track in self.tracks.items():
            best_iou = self.iou_threshold
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                iou = self.calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx != -1:
                det = detections[best_det_idx]
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['class_name'] = det['class_name']
                used_detections.add(best_det_idx)
                
                results.append({
                    'track_id': track_id,
                    'bbox': det['bbox'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence']
                })
            else:
                track['age'] += 1
                if track['age'] <= self.max_age:
                    results.append({
                        'track_id': track_id,
                        'bbox': track['bbox'],
                        'class_name': track['class_name'],
                        'confidence': 0.5
                    })
                else:
                    del self.tracks[track_id]
        
        # Новые треки
        for i, det in enumerate(detections):
            if i not in used_detections:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'age': 0,
                    'class_name': det['class_name']
                }
                
                results.append({
                    'track_id': track_id,
                    'bbox': det['bbox'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence']
                })
        
        return results
    
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)

class TrailDrawer:
    def __init__(self, max_length=30):
        self.trails = defaultdict(lambda: deque(maxlen=max_length))
        self.colors = {}
    
    def update(self, tracks, frame_width, frame_height):
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            # Центр bbox с проверкой границ
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Ограничиваем координаты рамкой видео
            center_x = max(0, min(center_x, frame_width))
            center_y = max(0, min(center_y, frame_height))
            
            self.trails[track_id].append((center_x, center_y))
            
            if track_id not in self.colors:
                np.random.seed(track_id)
                self.colors[track_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
    
    def draw(self, frame):
        for track_id, points in self.trails.items():
            if len(points) > 1:
                color = self.colors[track_id]
                
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    
                    # Проверяем что точки в пределах кадра
                    if (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0] and
                        0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]):
                        
                        # Плавное уменьшение толщины
                        alpha = i / len(points)
                        thickness = max(2, int(4 * alpha))
                        cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame

def calculate_font_scale(frame_width):
    """Адаптивный размер шрифта в зависимости от разрешения"""
    if frame_width >= 3840:  # 4K
        return 1.5
    elif frame_width >= 1920:  # Full HD
        return 1.0
    elif frame_width >= 1280:  # HD
        return 0.8
    else:
        return 0.6

def main():
    print("=" * 70)
    print("YOLOv3 ДЕТЕКТОР - ИСПРАВЛЕННОЕ МАСШТАБИРОВАНИЕ")
    print("=" * 70)
    
    # Параметры
    VIDEO_PATH = "C:\\Users\\admin\\Downloads\\car-traffic.mp4"
    CONF_THRESHOLD = 0.5
    MAX_AGE = 30
    IOU_THRESHOLD = 0.3
    SKIP_FRAMES = 2  # Пропускать 2 из 3 кадров для ускорения
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Видео не найдено: {VIDEO_PATH}")
        return
    
    # Инициализация
    print("1. Загрузка детектора...")
    detector = YOLOv3Detector(conf_thresh=CONF_THRESHOLD)
    
    print("2. Инициализация трекера...")
    tracker = SimpleTracker(max_age=MAX_AGE, iou_threshold=IOU_THRESHOLD)
    trail_drawer = TrailDrawer(max_length=50)
    
    print("3. Открытие видео...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return
    
    # Оригинальные параметры видео
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Оригинальный размер: {orig_width}x{orig_height}")
    print(f"   Для обработки будет использоваться: 1280x720")
    print(f"   FPS: {fps:.1f}")
    print(f"   Порог уверенности: {CONF_THRESHOLD}")
    print(f"   IoU порог: {IOU_THRESHOLD}")
    print("-" * 70)
    
    # Размер для обработки (фиксированный для стабильности)
    PROCESS_WIDTH = 1280
    PROCESS_HEIGHT = 720
    
    # Окно для отображения
    cv2.namedWindow('YOLOv3 Tracker', cv2.WINDOW_NORMAL)
    display_width = min(1920, orig_width)
    display_height = min(1080, orig_height)
    cv2.resizeWindow('YOLOv3 Tracker', display_width, display_height)
    
    # Для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_result.mp4', fourcc, fps, (orig_width, orig_height))
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print("4. Начинаю обработку...")
    print("   Нажмите 'q' для выхода, 's' для сохранения кадра")
    print("   Нажмите '+' для увеличения порога, '-' для уменьшения")
    print("-" * 70)
    
    current_conf_threshold = CONF_THRESHOLD
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Конец видео")
                break
            
            frame_count += 1
            
            # Пропуск кадров для ускорения
            if frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
            processed_count += 1
            
            # Сохраняем оригинальный кадр для отрисовки
            original_frame = frame.copy()
            
            # Уменьшаем кадр для обработки
            processed_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # Детекция (передаем оригинальный размер для масштабирования)
            detections = detector.detect(processed_frame, (orig_width, orig_height))
            
            # Трекинг
            tracks = tracker.update(detections)
            
            # Обновляем шлейфы с оригинальными размерами
            trail_drawer.update(tracks, orig_width, orig_height)
            
            # Отрисовка на оригинальном кадре
            display_frame = original_frame.copy()
            
            # 1. Рисуем шлейфы
            display_frame = trail_drawer.draw(display_frame)
            
            # 2. Адаптивный размер шрифта
            font_scale = calculate_font_scale(orig_width)
            font_thickness = max(2, int(font_scale * 2))
            
            # 3. Рисуем bounding boxes
            cars_count = 0
            people_count = 0
            other_count = 0
            
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ограничиваем координаты рамкой
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_width - 1, x2)
                y2 = min(orig_height - 1, y2)
                
                # Пропускаем если bbox слишком маленький или невалидный
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                # Цвет по ID трека
                track_id = track['track_id']
                np.random.seed(track_id)
                color = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
                
                # Класс объекта
                class_name = track['class_name']
                
                # Счетчики
                if class_name in ['car', 'bus', 'truck']:
                    cars_count += 1
                    box_color = (0, 0, 255)  # Красный для машин
                elif class_name == 'person':
                    people_count += 1
                    box_color = (0, 255, 0)  # Зеленый для людей
                elif class_name == 'motorcycle' or class_name == 'bicycle':
                    other_count += 1
                    box_color = (255, 0, 0)  # Синий для транспорта
                else:
                    other_count += 1
                    box_color = color
                
                # Толщина рамки адаптивная
                box_thickness = max(2, int(3 * (orig_width / 1920)))
                
                # Рисуем bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                
                # Подпись с адаптивным размером
                label = f"{class_name} ID:{track_id}"
                
                # Размер текста
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Рисуем фон для текста
                cv2.rectangle(display_frame,
                            (x1, y1 - label_height - baseline - 5),
                            (x1 + label_width, y1),
                            box_color, -1)
                
                # Рисуем текст
                cv2.putText(display_frame, label,
                          (x1, y1 - baseline - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                          (255, 255, 255), font_thickness)
            
            # 4. Панель статистики
            elapsed = time.time() - start_time
            current_fps = processed_count / elapsed if elapsed > 0 else 0
            
            # Темный фон для статистики
            stats_height = 130
            cv2.rectangle(display_frame, (0, 0), (700, stats_height), (30, 30, 30), -1)
            cv2.rectangle(display_frame, (0, 0), (700, stats_height), (0, 200, 0), 3)
            
            # Статистика
            stats_lines = [
                f"Видео: {os.path.basename(VIDEO_PATH)}",
                f"Кадр: {frame_count} | Обработано: {processed_count}",
                f"FPS: {current_fps:.1f}",
                f"Машин: {cars_count} | Людей: {people_count} | Других: {other_count}",
                f"Порог уверенности: {current_conf_threshold:.2f} (Используйте +/-)",
                f"Max Age: {MAX_AGE} | IoU: {IOU_THRESHOLD}",
                "Управление: 'q'-выход 's'-сохранить '+/-'-порог"
            ]
            
            # Адаптивный размер шрифта для статистики
            stats_font_scale = max(0.5, min(0.8, 1920 / orig_width))
            stats_thickness = max(1, int(stats_font_scale * 2))
            
            for i, line in enumerate(stats_lines):
                y_pos = 25 + i * 20
                cv2.putText(display_frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale,
                           (220, 255, 220), stats_thickness)
            
            # 5. Информация о масштабе (в правом нижнем углу)
            scale_info = f"Масштаб: 1:{orig_width//PROCESS_WIDTH}"
            cv2.putText(display_frame, scale_info, 
                       (orig_width - 250, orig_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
            
            # Сохраняем результат
            out.write(display_frame)
            
            # Показываем (уменьшаем для отображения если нужно)
            if orig_width > 1920 or orig_height > 1080:
                display_resized = cv2.resize(display_frame, (display_width, display_height))
                cv2.imshow('YOLOv3 Tracker', display_resized)
            else:
                cv2.imshow('YOLOv3 Tracker', display_frame)
            
            # Управление
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⚠️  Остановлено пользователем")
                break
            elif key == ord('s'):
                filename = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"\n📸 Сохранен кадр: {filename}")
            elif key == ord('+'):
                current_conf_threshold = min(0.9, current_conf_threshold + 0.05)
                detector.conf_thresh = current_conf_threshold
                print(f"\n🔺 Порог увеличен: {current_conf_threshold:.2f}")
            elif key == ord('-'):
                current_conf_threshold = max(0.1, current_conf_threshold - 0.05)
                detector.conf_thresh = current_conf_threshold
                print(f"\n🔻 Порог уменьшен: {current_conf_threshold:.2f}")
            
            # Прогресс
            if processed_count % 10 == 0:
                print(f"Кадр {frame_count} | FPS: {current_fps:.1f} | Машин: {cars_count} | Людей: {people_count}", end='\r')
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано (Ctrl+C)")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("СТАТИСТИКА ОБРАБОТКИ")
        print("=" * 70)
        print(f"Всего кадров: {frame_count}")
        print(f"Обработано кадров: {processed_count}")
        print(f"Общее время: {time.time() - start_time:.1f} сек")
        print(f"Средний FPS обработки: {processed_count/(time.time() - start_time):.1f}")
        print(f"Результат сохранен в: output_result.mp4")
        print("=" * 70)

if __name__ == "__main__":
    main()