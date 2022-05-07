import pickle as pkl

import numpy as np
import torch
import torchmetrics
from ISR.models import RDN
from PIL import Image
from tqdm import tqdm

# uses https://github.com/idealo/image-super-resolution
# to install use pip install isr --no-deps


def predict(file_name, folder_name, model, metric, result_arr):
    lr_img = np.array(Image.open(f"data/scaled/{folder_name}/{file_name}"))
    real_img = np.array(Image.open(f"data/processed/{folder_name}/{file_name}"))
    prediction = model.predict(lr_img)
    result_arr.append(metric(torch.tensor(prediction), torch.tensor(real_img)))


with open("data/cat_val.pkl", "rb") as f:
    names = pkl.load(f)

rdn = RDN(weights="psnr-small")
psnr = torchmetrics.PeakSignalNoiseRatio()
cat_psnr_values = []
dog_psnr_values = []

progress_bar = tqdm(range(len(names)), ascii=False, ncols=75)
for i in progress_bar:
    predict(names[i], "Cat", rdn, psnr, cat_psnr_values)
    predict(names[i], "Dog", rdn, psnr, dog_psnr_values)
    progress_bar.set_postfix(
        {"PSNR cats": np.mean(cat_psnr_values), "PSNR_dogs": np.mean(dog_psnr_values)}
    )


print("PeakSignalNoiseRatio for Cats prediction: %s" % str(np.mean(cat_psnr_values)))
print("PeakSignalNoiseRatio for Dogs prediction: %s" % str(np.mean(dog_psnr_values)))
