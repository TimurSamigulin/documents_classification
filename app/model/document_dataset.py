import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from typing import List
from transformers import LayoutLMv3Processor


class DocumentClassificationDataset(Dataset):
    
    def __init__(self, dataframe, doc_classes: List, processor: LayoutLMv3Processor):
        self.data = dataframe
        self.doc_classes = doc_classes
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def scale_bbox(self, box: List[int], width_scale: float, height_scale: float) -> List[int]:
        return [
            int(box[0] * width_scale),
            int(box[1] * height_scale),
            int(box[2] * width_scale),
            int(box[3] * height_scale)
        ]

    def __getitem__(self, item):
        image_path = self.data.iloc[item]['image_path']
        json_path = self.data.iloc[item]['ocr_path']
        category = self.data.iloc[item]['category']

        
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        with open(json_path, "r") as f:
            ocr_result = json.load(f)
        
        for row in ocr_result:
            width = max(width, row["bounding_box"][0], row["bounding_box"][2])
            height = max(height, row["bounding_box"][1], row["bounding_box"][3])
            
        width_scale = 1000 / width
        height_scale = 1000 / height
        
        words = []
        boxes = []
        
        for row in ocr_result:
            boxes_sc = self.scale_bbox(row["bounding_box"], width_scale, height_scale)
            for i in range(len(boxes_sc)):
                if boxes_sc[i] > 1000:
                    boxes_sc[i] = 1000
                elif boxes_sc[i] < 0:
                    boxes_sc[i] = 0
            boxes.append(boxes_sc)
            words.append(row["word"])

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        label = self.doc_classes.index(category)
                
        try:    
            return dict(
                input_ids = encoding["input_ids"].flatten(),
                attention_mask = encoding["attention_mask"].flatten(),
                bbox = encoding["bbox"].flatten(end_dim=1),
                pixel_values = encoding["pixel_values"].flatten(end_dim=1),
                labels = torch.tensor(label, dtype=torch.long)
            )
        except:
            print(image_path)