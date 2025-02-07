import io
import json
from typing import List

import torch
import open_clip
from fastapi import FastAPI, UploadFile, File, Depends
from PIL import Image
from pydantic import BaseModel

# if you're using a silicon Mac, you might see faster performance using mps, otherwise set MAC = False
MAC = True

# Initialize FastAPI
app = FastAPI()

# Load model and tokenizer
device = "mps" if MAC else "cpu"

print(f"[INFO] Loading CLIP to {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("[INFO] CLIP is ready!")


# Define request payload
class CLIPRequest(BaseModel):
    cls_options: List[str]


# Define CLIP endpoint
@app.post("/clip_predict")
async def clip_predict(request: CLIPRequest = Depends(), file: UploadFile = File(...)):
    with torch.no_grad():
        request_object_content = await file.read()
        pil_img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        image = preprocess(pil_img).unsqueeze(0).to(device)

        # read classification options from request
        cls_options = request.cls_options
        text = tokenizer(cls_options).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # output as json
        data_dict = {
            label: text_probs[0][i].item()
            for i, label in enumerate(cls_options)
        }

    return data_dict