# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from torchvision import transforms
import requests
import json
from timm.models import create_model

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: str = Input(description="URL of the input image"),
        model: str = Input(description="Model to be used"),
            ) -> str:
        """Run a single prediction on the model"""
        
        input_image = Image.open(requests.get(image, stream=True).raw)
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        input_tensor = preprocess(input_image)

        model = create_model(
            model,
            num_classes=1000,
            in_chans=3,
            pretrained=True,
            checkpoint_path='')


        input_batch = input_tensor.unsqueeze(0) 
        model.eval()

        with torch.no_grad():
            output = model(input_batch)        
                
        output = torch.nn.functional.softmax(output[0], dim=0)
        confidence, index = torch.max(output, 0)

        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        res = {}
        res["predicted"] = classes[index.item()]
        res["confidence"] = str(confidence.item())
        json_object = json.dumps(res)
        return json_object
        
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
