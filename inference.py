import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from model import SimCLR, LinearClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


def load_model():
    encoder = SimCLR().encoder
    classifier = LinearClassifier()

    model = nn.Sequential(encoder, classifier)
    model.load_state_dict(torch.load("checkpoints/final_model.pth", map_location=device))

    model.to(device)
    model.eval()

    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3444, 0.3803, 0.4078],
            std=[0.2034, 0.1365, 0.1148]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    return image


def predict(image_path):
    model = load_model()
    image = preprocess(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], conf.item()


if __name__ == "__main__":
    predict("test.jpg")