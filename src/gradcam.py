# Generate a Grad-CAM attention heatmap for one input image.
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/best.pt'); p.add_argument('--image',required=True); p.add_argument('--output',default='artifacts/gradcam.png'); a=p.parse_args()
    device=torch.device('cpu'); ck=torch.load(a.checkpoint,map_location=device); model=resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,10); model.load_state_dict(ck['model']); model.eval()
    activations=[]; gradients=[]
    target=model.layer4[-1].conv2
    target.register_forward_hook(lambda m,i,o: activations.append(o))
    target.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0]))
    image=Image.open(a.image).convert('RGB'); tf=transforms.Compose([transforms.Grayscale(3),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); x=tf(image).unsqueeze(0)
    logits=model(x); cls=int(logits.argmax(1)); model.zero_grad(); logits[0,cls].backward(); weight=gradients[0].mean(dim=(2,3),keepdim=True); cam=torch.relu((weight*activations[0]).sum(1,keepdim=True)); cam=torch.nn.functional.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0]; cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    heat=plt.get_cmap('jet')(cam.detach().numpy())[:,:,:3]; base=ImageOps.fit(image,(224,224),method=Image.Resampling.LANCZOS).convert('L').convert('RGB'); base_array=np.asarray(base,dtype=np.float32)/255.0; overlay=np.clip(.55*heat+.45*base_array,0,1); Path(a.output).parent.mkdir(parents=True,exist_ok=True); plt.imsave(a.output,overlay); print(f'class={CLASSES[cls]} output={a.output}')
if __name__=='__main__': main()

