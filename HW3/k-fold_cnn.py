#!/usr/bin/env python
# coding: utf-8

# # HW3 Image Classification
# ## We strongly recommend that you run with Kaggle for this homework
# https://www.kaggle.com/c/ml2022spring-hw3b/code?competitionId=34954&sortBy=dateCreated
# * Simple : 0.50099
# * Medium : 0.73207 Training Augmentation + Train Longer
# * Strong : 0.81872 Training Augmentation + Model Design + Train Looonger (+ Cross Validation + Ensemble)
# * Boss : 0.88446 Training Augmentation + Model Design +Test Time Augmentation + Train Looonger (+ Cross Validation + Ensemble) 

# # Get Data
# Notes: if the links are dead, you can download the data directly from Kaggle and upload it to the workspace, or you can use the Kaggle API to directly download the data into colab.
# 

# In[1]:


# ! kaggle competitions download -c ml2022spring-hw3b
# ! unzip ml2022spring-hw3b.zip


# In[2]:
# from IPython import get_ipython

# get_ipython().system('! nvidia-smi')


# # Training

# In[3]:


_exp_name = "CNN"


# In[4]:


# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# k-fold 
from sklearn.model_selection import KFold


# In[5]:


myseed = 0  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# In[4]:


# get_ipython().system("jupyter nbconvert --to script 'k-fold.ipynb'")1


# ## **Transforms**
# Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
# 
# Please refer to PyTorch official website for details about different transforms.

# In[6]:


# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # You need to add some transforms here.
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ToTensor() should be the last one of the transforms.
])

# ## **Datasets**
# The data is labelled by the name, so we load images and label while calling '__getitem__'

# In[7]:


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files = None):
        super(FoodDataset).__init__()
        if files != None:
            self.files = files
        else:
            self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        print(f"One sample", self.files[0])
        print(f"number of samples: {len(self.files)}")
        self.transform = tfm
  
    def __len__(self):
        len(self.files)
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im, label


# In[8]:


from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # [3, 224, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(),

            nn.Conv2d(32, 64, 3), # [32, 224, 224]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(),

            nn.Conv2d(64, 128, 3), # [64, 111, 111]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(),

            nn.Conv2d(128, 256, 3), # [128, 54, 54]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(), 

            nn.Conv2d(256, 512, 3), # [256, 26, 26]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(),
            
            nn.Conv2d(512, 512, 3), # [512, 12, 12]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # nn.Dropout(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*5*5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# In[10]:


batch_size = 32
_dataset_dir = "./food11"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
# train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
# valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)

# Define the k-fold cross validator
k = 5           # split data to 5 folds
k_fold = KFold(n_splits=k, shuffle=True)
files = [os.path.join(_dataset_dir, "training", x) for x in os.listdir(_dataset_dir + "/training") if x.endswith(".jpg")]\
        + [os.path.join(_dataset_dir, "validation", x) for x in os.listdir(_dataset_dir + "/validation") if x.endswith(".jpg")]
data_set = FoodDataset(None, tfm=train_tfm, files=files)

# In[11]:
# 10 epochs to warm up and then every 10 steps to decrease lr
lr_lambda = lambda epoch: (2 ** epoch) * 1e-3 if epoch <= 10 else (0.6 ** (epoch // 10 - 1))
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
total_acc = 0.0
# The number of training epochs and patience.
n_epochs = 150
patience = 300 # If no improvement in 'patience' epochs, early stop

for fold, (train_set_id, valid_set_id) in enumerate(k_fold.split(data_set)):
    # Initialize a model, and put it on the device specified.
    model = CNN().to(device)

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5) 
    # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Initialize trackers, these are not parameters and should not be changed
    best_acc = 0
    stale = 0

    # take a subset of train_set and valid_set
    train_set, valid_set = Subset(data_set, train_set_id), Subset(data_set, valid_set_id)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    
    for epoch in range(1, n_epochs+1):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in train_loader:
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        # print(optimizer.param_groups[0]['lr'])
        scheduler2.step()
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[FOLD: {fold+1}/{k} | Train: {epoch:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        
        
        # Iterate the validation set by batches.
        for batch in valid_loader:

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt", "a") as f:
                f.write(f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best\n")
            print(f"[FOLD: {fold+1}/{k} | Valid: {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt", "a") as f:
                f.write(f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            torch.save(model.state_dict(), f"{_exp_name}_fold_{fold}.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            # print(f"Best model found at epoch {epoch}, acc: {best_acc:.5f} saving model...")
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
    total_acc += best_acc


# In[ ]:


print(f"Average acc = {total_acc/k:.5f}")


# In[ ]:


file = sorted([os.path.join(_dataset_dir, "test", x) for x in os.listdir(_dataset_dir + "/test") if x.endswith(".jpg")])
test_set = FoodDataset(None, tfm=test_tfm, files=file)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# # Testing and generate prediction CSV

# In[ ]:



device = "cuda" if torch.cuda.is_available() else "cpu"
model0 = CNN().to(device)
model0.load_state_dict(torch.load(f"{_exp_name}_fold_0.ckpt"))
model0.eval()

model1 = CNN().to(device)
model1.load_state_dict(torch.load(f"{_exp_name}_fold_1.ckpt"))
model1.eval()

model2 = CNN().to(device)
model2.load_state_dict(torch.load(f"{_exp_name}_fold_2.ckpt"))
model2.eval()

model3 = CNN().to(device)
model3.load_state_dict(torch.load(f"{_exp_name}_fold_3.ckpt"))
model3.eval()

model4 = CNN().to(device)
model4.load_state_dict(torch.load(f"{_exp_name}_fold_4.ckpt"))
model4.eval()

prediction = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        
        test0 = model0(data)
        test1 = model1(data)
        test2 = model2(data)
        test3 = model3(data)
        test4 = model4(data)
        test_pred = (test0 + test1 + test2 + test3 + test4) / k
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


# In[ ]:


#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission_cnn.csv",index = False)
print('Finish preficting!')
