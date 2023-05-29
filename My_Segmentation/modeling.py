#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sem_seg_dataset import SegmentationDataset
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt
import time
# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"I am using {DEVICE}")
# %% Hyperparameters
EPOCHS = 10
BS = 4


# %%
train_ds = SegmentationDataset(path_name="data/train")
train_data_loader = DataLoader(train_ds,batch_size=BS,shuffle=True)
val_ds = SegmentationDataset(path_name="data/val")
val_data_loader = DataLoader(val_ds,batch_size=BS,shuffle=True)


# %%
model = smp.FPN(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    classes=6,
    activation="softmax",
 
)
model.to(DEVICE)
# %%
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

#%% TRAIN MODEL
criterion = nn.CrossEntropyLoss()
# criterion = smp.losses.DiceLoss(mode='multiclass')
train_losses, val_losses = [], []

#%%
start = time.time()
for e in range(EPOCHS):
    epoch_time = time.time()
    model.train()
    running_train_loss, running_val_loss = 0, 0
    for i, data in enumerate(train_data_loader):
        #training phase
        image_i, mask_i = data
        image = image_i.to(DEVICE)
        mask = mask_i.to(DEVICE)
        
        # reset gradients
        optimizer.zero_grad() 
        #forward
        output = model(image.float())
        
        # calc losses
        train_loss = criterion(output.float(), mask.long())

        # back propagation
        train_loss.backward()
        optimizer.step() #update weight          
        
        running_train_loss += train_loss.item()
    train_losses.append(running_train_loss) 
    
    # validation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_data_loader):
            image_i, mask_i = data
            image = image_i.to(DEVICE)
            mask = mask_i.to(DEVICE)
            #forward
            output = model(image.float())
            # calc losses
            val_loss = criterion(output.float(), mask.long())
            running_val_loss += val_loss.item()
    val_losses.append(running_val_loss) 
    
    epoch_time = time.time() - epoch_time
    print(f"Epoch: {e}: Train Loss: {np.median(running_train_loss)}, Val Loss: {np.median(running_val_loss)}, time usage: {epoch_time}")
total_time = time.time() - start
print(f"Total time usage: {total_time}")2
#%% TRAIN LOSS
sns.lineplot(x = range(len(train_losses)), y= train_losses).set(title='Train Loss')
plt.show()
sns.lineplot(x = range(len(train_losses)), y= val_losses).set(title='Validation Loss')
plt.show()

# %% save model
torch.save(model.state_dict(), f'models/FPN_epochs_{EPOCHS}_crossentropy_state_dict.pth')