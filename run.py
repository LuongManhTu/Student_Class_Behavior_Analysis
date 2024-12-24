import argparse
from students_detector import DinoDetector, YoloDetector
from behavior_classifier import Classifier
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy as np
import cv2
import time
import torch

def process_video(input_path, output_path, detector, classifier, tracker):
    start_time = time.time()
    video_cap = cv2.VideoCapture(input_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    track_colors = {}

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # Detector
        results = detector(frame) # ( [x,y,w,h], confidence, detection_class )
        
        # Tracker
        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            if track_id not in track_colors:
                track_colors[track_id] = tuple(np.random.randint(0, 255, size=3).tolist())
            color = track_colors[track_id]

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            cropped_image = frame[ymin : ymax, xmin : xmax]
            action, _ = classifier(cropped_image)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), color, -1)
            cv2.putText(frame, str(track_id) + "-" + action, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        
    end_time = time.time()
    process_time = end_time - start_time
    print(f"Finished processing video! Time: {process_time}")
    video_cap.release()
    writer.release()
    
    return process_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCB Analysis")
    
    parser.add_argument('--detector', type=str, default="YOLO", choices=["DINO", "YOLO"])
    parser.add_argument('--device', type=str, default="cpu", choices=["cuda", "cpu"])
    
    parser.add_argument('--dino_config_path', type=str, default="./groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--dino_weight_path', type=str, default="./weights/groundingdino_swint_ogc.pth")
    parser.add_argument('--yolo_weight_path', type=str, default="./weights/yolov8_best.pt")
    parser.add_argument('--resnet_weight_path', type=str, default="./weights/resnet_fit.pth")
    
    args = parser.parse_args()

    if args.detector == "DINO":
        detector = DinoDetector(args.dino_config_path, args.dino_weight_path, args.device)
        print(">>>> Loaded Detector: Grounding Dino!")
    else:
        detector = YoloDetector(args.yolo_weight_path, args.device)
        print(">>>> Loaded Detector: YOLOv8!")

    classifier = Classifier(args.resnet_weight_path, args.device)
    print(">>>> Loaded Classifier: Resnet!")

    if args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    tracker = DeepSort(max_age=50)

    input_path = "./test_videos/first.mp4"
    output_path = "./results/out_vis.mp4"

    process_video(input_path, output_path, detector, classifier, tracker)