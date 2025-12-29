import torch
import cv2
import numpy as np
from sort import *
from ultralytics import YOLO
import argparse
from collections import defaultdict, deque
from matplotlib import cm
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv3')
    parser.add_argument('--source', type=str, default="C:\\Users\\admin\\Downloads\\car-traffic.mp4")
    return parser.parse_args()

def colours(cls, cmap_name='hsv'):
    cmap = plt.get_cmap(cmap_name)
    return cmap(cls / 20.0)  # 20 colours for each class

def main():
    args = parse_arguments()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO('yolov3.pt').to(device)
    # model.conf = 42 # Установить ваш порог детектирования
    model.conf = 0.7
    model.eval()

    # add video
    try:
        source = int(args.source)
        cap = cv2.VideoCapture(source)
    except ValueError:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Не удалось открыть: {args.source}")
        return
    
    class_names = model.names
    
    mot_tracker = Sort() 
    mot_tracker.max_age = 1 # not remember object after 1 disappear
    mot_tracker.min_hits = 1
    mot_tracker.iou_threshold = 0.5
    
    use_tracker = True
    track_history = defaultdict(lambda: deque(maxlen=30))
    
    print("Управление: a/d - IoU, t - трекинг, q - выход")

    while True:
        ret, frame = cap.read()
        
        #convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # without this will be wrong photo colour
        
        # find objects
        results = model(frame, conf=model.conf) # results = model(frame)
        
        if len(results) > 0:
            preds = results[0].boxes.data.cpu().numpy()
        else:
            preds = np.zeros((0, 6))
        
        if use_tracker:
            if len(preds) > 0:
                tracked_preds = mot_tracker.update(preds)
                
                if len(tracked_preds) > 0:
                    # track hist update
                    for track in tracked_preds:
                        x1, y1, x2, y2, track_id = track[:5]
                        track_id = int(track_id)
                        center = (int((x1+x2)/2), int((y1+y2)/2))
                        track_history[track_id].append(center)
            else:
                tracked_preds = mot_tracker.update(np.zeros((0, 6)))
        
        # draw track
        if use_tracker and 'tracked_preds' in locals() and len(tracked_preds) > 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for track in tracked_preds:
                x1, y1, x2, y2, track_id = track[:5]
                x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                
                cls = 0
                if len(preds) > 0:
                    min_dist = float('inf')
                    track_center = ((x1+x2)/2, (y1+y2)/2)
                    
                    for pred in preds:
                        pred_center = ((pred[0]+pred[2])/2, (pred[1]+pred[3])/2)
                        dist = np.sqrt((track_center[0]-pred_center[0])**2 + 
                                     (track_center[1]-pred_center[1])**2)
                        if dist < min_dist and dist < 100:
                            min_dist = dist
                            cls = int(pred[5])
                
                # class colour chose
                color_tuple = colours(cls)
                clr = (int(color_tuple[0] * 255), 
                       int(color_tuple[1] * 255), 
                       int(color_tuple[2] * 255))
                
                # bounding box 
                cv2.rectangle(image, (x1, y1), (x2, y2), clr, 4)
                cv2.putText(image, f"{class_names[cls]} ID:{track_id}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, clr, 5)
                
                # track
                if track_id in track_history:
                    points = list(track_history[track_id])
                    for j in range(1, len(points)):
                        alpha = 0.3 + (j / len(points)) * 0.7
                        color_with_alpha = tuple(int(c * alpha) for c in clr)
                        cv2.line(image, points[j-1], points[j], color_with_alpha, 6)#4)
        else:
            # without track
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for pred in preds:
                x1, y1, x2, y2, conf, cls = pred
                x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
                color_tuple = colours(cls)
                clr = (int(color_tuple[0] * 255), 
                       int(color_tuple[1] * 255), 
                       int(color_tuple[2] * 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), clr, 4)
                cv2.putText(image, class_names[cls], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, clr, 5)
        
        # default inf
        info_text = f"Conf: {round(model.conf, 2)} | Tracks: {len(track_history)} | Tracker: {'ON' if use_tracker else 'OFF'} | In memory {mot_tracker.max_age}"
        cv2.putText(image, info_text, (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
        
        resized_image = cv2.resize(image, (800, 600))
        cv2.imshow("YOLOv3", resized_image)
        
        # react on buttons
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            model.conf = min(1.0, model.conf + 0.05)
            print(f"Порог увеличен: {round(model.conf, 2)}")
        elif key == ord('d'):
            model.conf = max(0.05, model.conf - 0.05)
            print(f"Порог уменьшен: {round(model.conf, 2)}")
        elif key == ord('w'):
            mot_tracker.max_age += 1
            print(f"Время жизни увеличено: {mot_tracker.max_age}")
        elif key == ord('s'):
            mot_tracker.max_age = max(1, mot_tracker.max_age - 1)
            print(f"Время жизни уменьшено: {mot_tracker.max_age}")
        elif key == ord('t'):
            use_tracker = not use_tracker
            print(f"Трекинг: {'ВКЛ' if use_tracker else 'ВЫКЛ'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()