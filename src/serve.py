import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from dataset import get_transforms
from model import get_model

app = FastAPI()
model = None

CHECKPOINT_PATH = Path("/app/checkpoints/classifier_v1.pt")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


@app.on_event("startup")
def load_model():
    global model
    model = get_model(architecture="resnet18", num_classes=10)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "not_ready"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    return {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}