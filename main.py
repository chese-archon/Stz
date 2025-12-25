import torch
import sys
import cv2
import numpy as np
import argparse
import os
import time
import signal

# Импортируем оригинальный трекер из sort.py
from sort import Sort

# ==================== ФУНКЦИЯ ДЛЯ ЦВЕТОВ ====================
def get_color(obj_id, alpha=1.0):
    """
    Генерация цветов на основе ID объекта
    Используем детерминированный подход для постоянства цветов
    """
    np.random.seed(int(obj_id) % 32)
    color = np.random.rand(3) * 255
    return tuple([int(c * alpha) for c in color])

# ==================== ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ОКНА ====================
def is_window_closed(window_name):
    """
    Проверяет, закрыто ли окно OpenCV
    Возвращает True если окно закрыто
    """
    try:
        # Для OpenCV 4.x и 3.x
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        # Если окно уже уничтожено
        return True
    except AttributeError:
        # Для старых версий OpenCV
        try:
            return cv2.getWindowProperty(window_name, 0) < 0
        except:
            return True

# ==================== ОБРАБОТЧИК СИГНАЛОВ ====================
def signal_handler(sig, frame):
    """Обработчик Ctrl+C для graceful shutdown"""
    print('\n\nПолучен сигнал Ctrl+C, завершение работы...')
    sys.exit(0)

# ==================== МОДЕЛЬ YOLOv3 ====================
class YOLOv3Model:
    """Обертка для YOLOv3 в стиле PyTorch модели"""
    def __init__(self, config_path="data/yolov3.cfg", 
                 weights_path="data/yolov3.weights",
                 classes_path="data/coco.names"):
        
        # Загружаем классы COCO
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Загружаем модель YOLOv3 через OpenCV DNN
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Получаем выходные слои
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Параметры модели
        self.conf = 0.5  # Порог уверенности
        self.iou = 0.4   # IoU для NMS
        self.names = {i: name for i, name in enumerate(self.classes)}
        
        print(f"✓ YOLOv3 модель загружена ({len(self.classes)} классов)")
    
    def to(self, device):
        """Для совместимости с PyTorch API"""
        print(f"Модель на устройстве: {device}")
        return self
    
    def eval(self):
        """Режим оценки"""
        return self
    
    def __call__(self, image):
        """
        Выполняет детекцию объектов
        Возвращает детекции в формате [[x1, y1, x2, y2, conf, cls], ...]
        """
        height, width = image.shape[:2]
        
        # Создаем blob для YOLOv3
        blob = cv2.dnn.blobFromImage(
            image, 
            1/255.0,           # Масштабирование
            (416, 416),        # Размер для YOLOv3
            (0, 0, 0),         # Средние значения
            swapRB=True,       # Уже в RGB
            crop=False
        )
        
        # Пропускаем через сеть
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Обрабатываем результаты
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf:
                    # Координаты bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.iou)
        
        # Формируем результат в нужном формате для SORT
        results_list = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                cls = class_ids[i]
                # Формат для SORT: [x1, y1, x2, y2, score, class]
                results_list.append([x, y, x + w, y + h, conf, cls])
        
        # Создаем объект Results (для совместимости)
        class Results:
            def __init__(self, pred, image):
                self.pred = pred  # Список тензоров
                self.ims = [image]  # Список изображений
        
        # Конвертируем в тензоры PyTorch
        if results_list:
            pred_tensor = torch.tensor(results_list)
        else:
            pred_tensor = torch.zeros((0, 6))
        
        return Results([pred_tensor], image)

# ==================== КЛАСС ДЛЯ УПРАВЛЕНИЯ ТРАЕКТОРИЯМИ ====================
class TrajectoryManager:
    """Управляет треками и их траекториями для отрисовки шлейфа"""
    def __init__(self, max_history=50, fade_frames=30):
        """
        Args:
            max_history: максимальная длина траектории в кадрах
            fade_frames: количество кадров для полного исчезновения неактивных траекторий
        """
        self.trajectories = {}  # track_id -> {'points': [(x, y), ...], 'active': bool, 'inactive_frames': int}
        self.max_history = max_history
        self.fade_frames = fade_frames
    
    def update(self, trackers):
        """
        Обновляет траектории на основе текущих треков
        trackers: массив от SORT [[x1, y1, x2, y2, track_id, cls], ...]
        """
        current_ids = set()
        
        # Отмечаем все существующие треки как неактивные
        for track_id in self.trajectories:
            self.trajectories[track_id]['active'] = False
        
        if len(trackers) > 0:
            for tracker in trackers:
                if len(tracker) >= 6:
                    x1, y1, x2, y2, track_id, cls = tracker[:6]
                    track_id = int(track_id)
                    
                    # Вычисляем центр bounding box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    
                    # Создаем новую запись или обновляем существующую
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = {
                            'points': [],
                            'active': True,
                            'inactive_frames': 0,
                            'class': cls
                        }
                    else:
                        self.trajectories[track_id]['active'] = True
                        self.trajectories[track_id]['inactive_frames'] = 0
                    
                    # Добавляем точку в траекторию
                    self.trajectories[track_id]['points'].append((x_center, y_center))
                    
                    # Ограничиваем длину истории
                    if len(self.trajectories[track_id]['points']) > self.max_history:
                        self.trajectories[track_id]['points'].pop(0)
                    
                    current_ids.add(track_id)
        
        # Увеличиваем счетчик неактивных кадров для треков, которые не обновились
        for track_id in list(self.trajectories.keys()):
            if not self.trajectories[track_id]['active']:
                self.trajectories[track_id]['inactive_frames'] += 1
        
        # Удаляем очень старые треки (полностью исчезнувшие)
        track_ids_to_remove = []
        for track_id, data in self.trajectories.items():
            if data['inactive_frames'] > self.fade_frames * 2:  # В 2 раза больше времени фейда
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            del self.trajectories[track_id]
    
    def draw_trajectories(self, image, thickness=2, max_points=50):
        """Рисует шлейфы траекторий на изображении"""
        for track_id, data in self.trajectories.items():
            points = data['points']
            inactive_frames = data['inactive_frames']
            is_active = data['active']
            
            if len(points) < 2:
                continue
            
            # Получаем цвет для этого трека
            base_color = get_color(track_id, alpha=1.0)
            
            # Определяем прозрачность в зависимости от активности
            if is_active:
                alpha = 1.0  # Полностью видимый для активных
            else:
                # Плавное исчезновение для неактивных
                alpha = max(0.1, 1.0 - (inactive_frames / self.fade_frames))
            
            # Применяем прозрачность к цвету
            color = tuple([int(c * alpha) for c in base_color])
            
            # Берем только последние max_points точек
            recent_points = points[-max_points:] if len(points) > max_points else points
            
            # Рисуем линии между точками
            for i in range(1, len(recent_points)):
                pt1 = (int(recent_points[i-1][0]), int(recent_points[i-1][1]))
                pt2 = (int(recent_points[i][0]), int(recent_points[i][1]))
                
                # Пропускаем слишком далекие точки (возможно, ошибочные)
                dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                if dist < 100:  # Максимальное расстояние между точками
                    # Делаем сегменты дальше от текущей позиции более прозрачными
                    segment_index = i / len(recent_points)
                    if is_active:
                        segment_alpha = 0.3 + 0.7 * segment_index
                    else:
                        segment_alpha = alpha * (0.3 + 0.7 * segment_index)
                    
                    segment_color = tuple([int(c * segment_alpha) for c in base_color])
                    
                    # Толщина линии зависит от активности
                    segment_thickness = thickness if is_active else max(1, thickness - 1)
                    
                    cv2.line(image, pt1, pt2, segment_color, segment_thickness)
            
            # Рисуем последнюю точку (текущую позицию) для активных треков
            if is_active and len(recent_points) > 0:
                last_point = recent_points[-1]
                cv2.circle(image, (int(last_point[0]), int(last_point[1])), 
                           thickness + 2, color, -1)
        
        return image
    
    def get_active_count(self):
        """Возвращает количество активных треков"""
        return sum(1 for data in self.trajectories.values() if data['active'])
    
    def get_total_count(self):
        """Возвращает общее количество треков (активных + неактивных)"""
        return len(self.trajectories)

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    # Регистрация обработчика Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='YOLOv3 Object Detection with SORT Tracking')
    
    # 1. Устройство захвата (путь до видео, номер камеры, ip адрес)
    parser.add_argument('--source', type=str, 
                       default="C:\\Users\\admin\\Downloads\\car-traffic.mp4",
                       help='Путь к видеофайлу, номер камеры (0, 1, ...) или IP-адрес')
    
    # 2. Порог срабатывания детектора
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Порог уверенности детектора (0.0-1.0)')
    
    # 3. Время жизни объектов (когда трекер перестает быть активным)
    parser.add_argument('--max-age', type=int, default=30,
                       help='Максимальное количество кадров без обновления до удаления трека')
    
    # 4. Порог IoU для трекера
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                       help='Порог IoU для ассоциации детекций с треками (0.0-1.0)')
    
    # Дополнительные параметры из оригинального SORT
    parser.add_argument('--min-hits', type=int, default=3,
                       help='Минимальное количество детекций для инициализации трека')
    
    # Флаги управления
    parser.add_argument('--use-tracker', action='store_true', default=True,
                       help='Использовать трекинг объектов')
    parser.add_argument('--show-trajectories', action='store_true', default=True,
                       help='Показывать шлейфы траекторий')
    parser.add_argument('--trajectory-length', type=int, default=30,
                       help='Длина отображаемого шлейфа траектории (в кадрах)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLOv3 Object Detection with SORT Multi-Object Tracking")
    print("=" * 70)
    print(f"Источник: {args.source}")
    print(f"Порог детектора: {args.conf_threshold}")
    print(f"Время жизни треков (max_age): {args.max_age} кадров")
    print(f"Порог IoU: {args.iou_threshold}")
    print(f"Минимальные попадания (min_hits): {args.min_hits}")
    print("=" * 70)
    
    # Инициализация устройства
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Загрузка модели YOLOv3
    print("\nЗагрузка модели YOLOv3...")
    try:
        model = YOLOv3Model(
            config_path="data/yolov3.cfg",
            weights_path="data/yolov3.weights",
            classes_path="data/coco.names"
        )
        model.conf = args.conf_threshold
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        print("Убедитесь, что файлы YOLOv3 находятся в папке 'data/':")
        print("  - yolov3.cfg")
        print("  - yolov3.weights")
        print("  - coco.names")
        return
    
    # Определяем тип источника
    source = args.source
    if source.isdigit():  # Номер камеры
        source = int(source)
        print(f"Используется камера #{source}")
    elif source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        print(f"Используется IP-камера/поток: {source}")
    else:
        print(f"Используется видеофайл: {source}")
    
    # Открытие видеофайла/камеры
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {args.source}")
        return
    
    # Получение параметров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nПараметры источника:")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    
    # Инициализация оригинального SORT трекера
    mot_tracker = Sort(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )
    
    # Менеджер траекторий для отрисовки шлейфа
    trajectory_manager = TrajectoryManager(max_history=args.trajectory_length)
    
    use_tracker = args.use_tracker
    show_trajectories = args.show_trajectories
    
    # Создание окна для отображения
    window_name = 'YOLOv3 Object Tracking'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))
    
    print("\n" + "=" * 70)
    print("УПРАВЛЕНИЕ:")
    print("  'q' или ESC - выход")
    print("  't' - вкл/выкл трекинг")
    print("  'p' - вкл/выкл отображение траекторий")
    print("  '+'/'-' - увеличить/уменьшить порог уверенности")
    print("  's' - сохранить текущий кадр")
    print("  ПРОБЕЛ - пауза/продолжить")
    print("  Ctrl+C - аварийное завершение")
    print("=" * 70 + "\n")
    
    frame_count = 0
    total_fps = 0
    running = True
    
    try:
        while running:
            # ПРОВЕРКА 1: Окно закрыто крестиком?
            if is_window_closed(window_name):
                print("\nОкно закрыто пользователем (крестик)")
                running = False
                break
            
            # ПРОВЕРКА 2: Чтение кадра
            ret, frame = cap.read()
            if not ret:
                print("\nКонец видео или ошибка чтения кадра")
                running = False
                break
            
            frame_count += 1
            
            # Конвертация BGR → RGB для модели
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Измерение времени детекции
            start_time = time.time()
            
            # Детекция объектов
            results = model(frame_rgb)
            
            # Получение предсказаний в формате numpy
            preds = results.pred[0].detach().cpu().numpy()
            
            detection_time = time.time() - start_time
            
            # Трекинг с оригинальным SORT
            tracking_time = 0
            if use_tracker and len(preds) > 0:
                # SORT ожидает формат: [[x1, y1, x2, y2, score, class], ...]
                # Наши данные уже в этом формате
                
                start_track = time.time()
                tracked_preds = mot_tracker.update(preds)
                tracking_time = time.time() - start_track
                
                # SORT возвращает: [[x1, y1, x2, y2, track_id, class], ...]
                # Конвертируем в наш формат: [[x1, y1, x2, y2, track_id, conf, class], ...]
                results_list = []
                if len(tracked_preds) > 0:
                    for track in tracked_preds:
                        x1, y1, x2, y2, track_id, cls = track[:6]
                        # Находим confidence из оригинальных детекций (приблизительно)
                        conf = 0.8  # Значение по умолчанию
                        # Можно улучшить: найти ближайшую детекцию и взять её confidence
                        results_list.append([x1, y1, x2, y2, track_id, conf, cls])
                
                preds = np.array(results_list) if results_list else np.zeros((0, 7))
                
                # Обновляем траектории для отрисовки шлейфа
                if show_trajectories:
                    trajectory_manager.update(tracked_preds)
            else:
                # Без трекера - просто форматируем данные
                if len(preds) > 0:
                    temp_preds = []
                    for pred in preds:
                        if len(pred) >= 6:
                            # track_id = -1 означает отсутствие трекинга
                            temp_preds.append([*pred[:6], -1])
                    preds = np.array(temp_preds) if temp_preds else np.zeros((0, 7))
                else:
                    preds = np.zeros((0, 7))
            
            # Конвертация обратно в BGR для отображения
            display_image = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
            
            # Рисуем траектории (шлейфы)
            if use_tracker and show_trajectories:
                display_image = trajectory_manager.draw_trajectories(display_image, thickness=6)
            
            # Отрисовка bounding boxes
            for pred in preds:
                if len(pred) >= 7:
                    # Формат с трекингом: [x1, y1, x2, y2, track_id, conf, cls]
                    x1, y1, x2, y2, track_id, conf, cls = pred[:7]
                    track_id = int(track_id)
                    cls = int(cls)
                elif len(pred) >= 6:
                    # Формат без трекинга: [x1, y1, x2, y2, conf, cls]
                    x1, y1, x2, y2, conf, cls = pred[:6]
                    track_id = -1
                    cls = int(cls)
                else:
                    continue
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Проверка координат
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width-1, x2), min(height-1, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Выбор цвета на основе track_id или class_id
                if track_id >= 0:
                    color_seed = track_id
                else:
                    color_seed = cls
                
                color = get_color(color_seed)
                
                # Имя класса
                class_name = model.names.get(cls, f'class_{cls}')
                
                # Отрисовка прямоугольника
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                
                # Подготовка текста
                font_scale = 1.5
                thickness = 3
                
                if use_tracker and track_id >= 0:
                    label = f"ID:{track_id} {class_name}"
                else:
                    label = f"{class_name} {conf:.2f}"
                
                # Размер текста
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Позиция текста
                text_y = max(y1, label_h + 5)
                
                # Фон для текста
                cv2.rectangle(display_image,
                             (x1, text_y - label_h - 5),
                             (x1 + label_w, text_y + 5),
                             color, -1)
                
                # Текст
                cv2.putText(display_image, label, 
                           (x1, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                           (255, 255, 255), thickness)
                
                # Для треков отображаем дополнительную информацию
                if use_tracker and track_id >= 0:
                    # Рисуем маленький кружок в центре объекта
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(display_image, (center_x, center_y), 3, (255, 255, 255), -1)
            
            # Добавление информации на кадр
            current_fps = 1.0 / (detection_time + tracking_time) if (detection_time + tracking_time) > 0 else 0
            total_fps += current_fps
            
            # Панель информации
            y_offset = 40
            line_height = 40 #25
            
            info_lines = [
                f"Кадр: {frame_count} | Объекты: {len(preds)}",
                f"Трекер: {'ВКЛ' if use_tracker else 'ВЫКЛ'} | Траектории: {'ВКЛ' if show_trajectories else 'ВЫКЛ'}",
                f"Порог: {model.conf:.2f} | Треков: {len(trajectory_manager.trajectories)}",
                f"Детекция: {detection_time*1000:.1f}мс | Трекинг: {tracking_time*1000:.1f}мс | FPS: {current_fps:.1f}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(display_image, line, (30, y_offset + i * line_height),
                           cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
            
            # Отображение изображения
            cv2.imshow(window_name, display_image)
            
            # ПРОВЕРКА 3: Обработка клавиш (30 мс для лучшей обработки событий окна)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' или ESC
                print("\nОстановлено пользователем (клавиша q/ESC)")
                running = False
                break
            elif key == ord('t'):  # Переключение трекера
                use_tracker = not use_tracker
                if use_tracker:
                    # Создаем новый трекер с текущими параметрами
                    mot_tracker = Sort(
                        max_age=args.max_age,
                        min_hits=args.min_hits,
                        iou_threshold=args.iou_threshold
                    )
                    trajectory_manager = TrajectoryManager(max_history=args.trajectory_length)
                    print(f"\nТрекер: ВКЛЮЧЕН (max_age={args.max_age}, iou={args.iou_threshold})")
                else:
                    print(f"\nТрекер: ВЫКЛЮЧЕН")
            elif key == ord('p'):  # Переключение отображения траекторий
                show_trajectories = not show_trajectories
                print(f"\nТраектории: {'ВКЛЮЧЕНЫ' if show_trajectories else 'ВЫКЛЮЧЕНЫ'}")
            elif key == ord('s'):  # Сохранение кадра
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{frame_count}_{timestamp}.jpg"
                cv2.imwrite(filename, display_image)
                print(f"\nКадр сохранен: {filename}")
            elif key == ord('+'):  # Увеличить порог уверенности
                model.conf = min(0.9, model.conf + 0.05)
                print(f"\nПорог уверенности увеличен до: {model.conf:.2f}")
            elif key == ord('-'):  # Уменьшить порог уверенности
                model.conf = max(0.1, model.conf - 0.05)
                print(f"\nПорог уверенности уменьшен до: {model.conf:.2f}")
            elif key == ord(' '):  # Пауза
                print("\nПауза. Нажмите ПРОБЕЛ для продолжения...")
                paused = True
                while paused and running:
                    # Проверяем не закрыто ли окно во время паузы
                    if is_window_closed(window_name):
                        print("\nОкно закрыто во время паузы")
                        paused = False
                        running = False
                        break
                    
                    key_pause = cv2.waitKey(30) & 0xFF
                    if key_pause == ord(' '):  # Снова пробел для продолжения
                        print("Продолжение...")
                        paused = False
                    elif key_pause == ord('s'):  # Сохранить кадр в паузе
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"frame_{frame_count}_paused_{timestamp}.jpg"
                        cv2.imwrite(filename, display_image)
                        print(f"Кадр сохранен: {filename}")
                    elif key_pause == 27 or key_pause == ord('q'):  # Выход из паузы
                        print("\nВыход из паузы с завершением программы")
                        paused = False
                        running = False
                        break
            
            # Прогресс каждые 30 кадров
            if frame_count % 30 == 0:
                avg_fps = total_fps / 30
                print(f"Кадр: {frame_count} | Объектов: {len(preds)} | FPS: {avg_fps:.1f}", end='\r')
                total_fps = 0
                
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\n\nПроизошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Всегда освобождаем ресурсы
        print("\n" + "=" * 70)
        print("ЗАВЕРШЕНИЕ РАБОТЫ...")
        print("=" * 70)
        
        # Закрываем видео
        cap.release()
        
        # Закрываем все окна OpenCV
        cv2.destroyAllWindows()
        
        # Дополнительный вызов для гарантии закрытия окон
        for i in range(5):
            cv2.waitKey(1)
        
        print(f"Всего обработано кадров: {frame_count}")
        print("Программа завершена.")
        print("=" * 70)

if __name__ == '__main__':
    main()