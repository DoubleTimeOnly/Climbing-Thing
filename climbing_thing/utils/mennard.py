from black import main
import torch
import clip
import numpy as np
from PIL import Image

MODELS = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

def colors_probs(images, prompts):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(MODELS[5], device=device)
 
    text = clip.tokenize(prompts).to(device)

    preprocessed_images = [preprocess(image).to(device) for image in images]

    batch = torch.stack(preprocessed_images, dim=0)

    with torch.no_grad():
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
    main()
