from typing import List

from black import main
import torch
import clip
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import Compose, CenterCrop, Resize, Normalize, InterpolationMode, ToTensor

MODELS = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

def colors_probs(images: List[np.ndarray], prompts: List[str]):
    """
    :param images: shape (N # Batch, Number, Timesteps, C, H, W)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(MODELS[5], device=device)
    n_px = model.visual.input_resolution
    preprocess = Compose([
            ToTensor(),
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # preprocessed_images = preprocess(images).to(device)
    batch = [preprocess(img).to(device) for img in images]
    batch = torch.stack(batch, dim=0)

    text = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(batch)
        # text_features = model.encode_text(text)

        logits_per_image, _ = model(batch, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs
  

def main():
    colors = ["purple", "white", "green", "yellow", "black", "orange", "blue"]
    prompts = colors
    # for color in colors:
    #     prompts += [f"{color} climbing hold", f"{color} climbing hold with white powder"]

    images = [
            Image.open(f"./climbing_thing/data/instance_images/test2/hold_{x}.png")
            for x in range(79)
        ]
    probs = colors_probs(images, prompts)
    for i, prob in enumerate(probs):
        print(f"{i}: ", [(prompts[i], prob[i]) for i in np.argsort(prob)[::-1]])

if __name__ == "__main__":
    import os
    from climbing_thing import ROOT_DIR
    os.chdir(os.path.join(ROOT_DIR, ".."))
    main()
