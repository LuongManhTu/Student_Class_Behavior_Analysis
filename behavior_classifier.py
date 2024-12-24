import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, WEIGHT_PATH, device):
        super(Classifier, self).__init__()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = torch.load(WEIGHT_PATH, map_location=device)
        self.device = device

        self.CLASS_NAMES = ['raising_hand', 'reading', 'sleeping', 'using_phone', 'writing']
        
    def forward(self, image):
        image = self.trans(image)
        image = image.to(self.device).unsqueeze(0)
        
        output = self.model(image)
        _, preds = torch.max(output, dim=1)
        
        softmax_scores = F.softmax(output, dim=1)
        score = softmax_scores[0][preds].item()

        return self.CLASS_NAMES[preds], score