#%% packages
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sem_seg_dataset import SegmentationDataset
import segmentation_models_pytorch as smp 
import torchmetrics
# %% Dataset and Dataloader
test_ds = SegmentationDataset(path_name="data/test")
test_data_loader = DataLoader(test_ds,batch_size=1,shuffle=True)
# %% Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%
model = smp.UNET(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    classes=6,
    activation="softmax",
 
)
model.to(DEVICE)
# %%
model.load_state_dict(torch.load("models/FPN_10_4_0.001_statedict.pth"))

# %%
pixel_acc = []
intersection_over_union = []
metric_iou = torchmetrics.JaccardIndex(num_classes=6,task="multiclass").to(DEVICE)
model.eval()
with torch.no_grad():
    for data in test_data_loader:
        inputs,outputs = data
        true = outputs.to(torch.float32).to(DEVICE)
        pred = model(inputs.to(DEVICE).float())
        predicted_class = torch.max(pred,1)[1]
        correct_pixels = torch.sum(predicted_class == true).item()
        total_pixels = true.shape[0]*true.shape[1]*true.shape[2]
        pixel_acc.append(correct_pixels/total_pixels)
        iou = metric_iou(predicted_class.float(),true).item()
        intersection_over_union.append(iou)
# %


# %%
print(f"Pixel Accuracy: {np.mean(pixel_acc)}")
print(f"Intersection over Union: {np.mean(intersection_over_union)}")
# %%
image_test,mask= next(iter(test_data_loader))
plt.imshow(image_test[0].permute(1,2,0))
# %%
