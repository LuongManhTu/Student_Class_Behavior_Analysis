from groundingdino.util.inference import load_model, load_image, predict
from ultralytics import YOLO
from helper import *
import torch.nn as nn

class DinoDetector(nn.Module):
    def __init__(self, CONFIG_PATH, WEIGHT_PATH, device):
        super(DinoDetector, self).__init__()
        self.TEXT_PROMPT = "single people"
        self.BOX_THRESHOLD = 0.3
        self.TEXT_THRESHOLD = 0.25

        self.device = device
        self.model = load_model(CONFIG_PATH, WEIGHT_PATH, device).to(device)

    def forward(self, frame):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        
        _, image = load_image_from_frame(frame)
        image = image.to(self.device)
        
        with torch.no_grad():
            boxes, scores, class_ids = predict(
                  model=self.model,
                  image=image,
                  caption=self.TEXT_PROMPT,
                  box_threshold=self.BOX_THRESHOLD,
                  text_threshold=self.TEXT_THRESHOLD
            )

        absolute_boxes = denormalize_boxes(boxes, image_width=frame_width, image_height=frame_height)
        absolute_boxes = post_process(absolute_boxes, scores)

        results = []
        for idx, box in enumerate(absolute_boxes):
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            results.append(([xmin, ymin, width, height], scores[idx], class_ids[idx]))  # Assume class_id = 0 for single person
    
        return results

class YoloDetector(nn.Module):
    def __init__(self, WEIGHT_PATH, device):
        super(YoloDetector, self).__init__()
        self.device = device
        self.model = YOLO(WEIGHT_PATH) #.to(device)
        self.model.to(device)
        
        self.CONFIDENCE_THRESHOLD = 0.3

    def forward(self, frame):

        print(">>>>>>>>>>>>>>>> BEFORE DEATH")
        
        detections = self.model(frame)[0]

        results = []
        for data in detections.boxes.data.tolist():
            confidence = data[4]
    
            if float(confidence) < self.CONFIDENCE_THRESHOLD:
                continue
    
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])

            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        return results
        