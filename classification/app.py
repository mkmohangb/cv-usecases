from flask import Flask, request, send_from_directory
import io
from PIL import Image
import torch
from torchvision import models
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.transforms import transforms
from urllib.request import urlopen

app = Flask(__name__)
model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
model.eval()

class_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_labels = urlopen(class_labels_url).read().decode('utf-8').split('\n')

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                ])

def predict(model, transform, image, class_labels):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    class_id = torch.argmax(output).item()
    class_name = class_labels[class_id]
    return class_name

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    image_file = request.files["file"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    return predict(model, transform, image, class_labels)
