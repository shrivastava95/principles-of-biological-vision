import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name='blip_caption', model_type='large_coco', is_eval=True, device=device)

def caption_blip(model, vis_processors, image):
    image = vis_processors['eval'](image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})
    return caption

caption_list = []

source_folder = 'natural images'
image_paths = [os.path.join(source_folder, image_path) for image_path in os.listdir(source_folder)]
image_paths.sort()
for image_path in os.listdir(source_folder):
    image_path = os.path.join(source_folder, image_path)
    image = Image.open(image_path)
    caption = caption_blip(model, vis_processors, image)
    caption_list.append(caption[0])

with open('caption_list.txt', 'w') as f:
    f.write('\n'.join(caption_list))