# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# k-fold 
from sklearn.model_selection import KFold
myseed = 0  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


class FoodDataset(Dataset):

    def __init__(self, path, tfm=transforms.ToTensor(), files = None):
        super(FoodDataset).__init__()
        # self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
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

from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2, 0)
        # [3, 224, 224]
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),   
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.cnn_layer8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512* 7* 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 224, 224]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)
        
        x2 = self.cnn_layer2(x1)
        x2 = torch.add(x2, x1) # add operation
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x3 = self.cnn_layer3(x2)
        
        x4 = self.cnn_layer4(x3)
        x4 = torch.add(x4, x3) # add operation
        x4 = self.relu(x4)
        
        x5 = self.cnn_layer5(x4)
        
        x6 = self.cnn_layer6(x5)
        x6 = torch.add(x6, x5) # add operation
        x6 = self.relu(x6)

        x7 = self.cnn_layer7(x6)

        x8 = self.cnn_layer8(x7)
        x8 = torch.add(x8, x7) # add operation
        x8 = self.relu(x8)

        # The extracted feature map must be flatten before going to fully-connected layers.
        xout = x8.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(xout)
        return xout

class myResnet18(nn.Module):
    def __init__(self) -> None:
        super(myResnet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)

        self.fc_layer = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 11)
        )
    def forward(self, x):
        x1 = self.resnet18(x)
        xout = self.fc_layer(x1)
        return xout


# Define the k-fold cross validator
k = 5           # split data to 5 folds
k_fold = KFold(n_splits=k, shuffle=True)
batch_size = 32
_dataset_dir = "./food11"
device = "cuda" if torch.cuda.is_available() else "cpu"

file = sorted([os.path.join(_dataset_dir, "test", x) for x in os.listdir(_dataset_dir + "/test") if x.endswith(".jpg")])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_set = FoodDataset(None, tfm=test_tfm, files=file)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print('Loading model...')
cnnmodel0 = CNN().to(device)
cnnmodel0.load_state_dict(torch.load(f"CNN_fold_0.ckpt"))
cnnmodel0.eval()

cnnmodel1 = CNN().to(device)
cnnmodel1.load_state_dict(torch.load(f"CNN_fold_1.ckpt"))
cnnmodel1.eval()

cnnmodel2 = CNN().to(device)
cnnmodel2.load_state_dict(torch.load(f"CNN_fold_2.ckpt"))
cnnmodel2.eval()

cnnmodel3 = CNN().to(device)
cnnmodel3.load_state_dict(torch.load(f"CNN_fold_3.ckpt"))
cnnmodel3.eval()

cnnmodel4 = CNN().to(device)
cnnmodel4.load_state_dict(torch.load(f"CNN_fold_4.ckpt"))
cnnmodel4.eval()

resmodel0 = Residual_Network().to(device)
resmodel0.load_state_dict(torch.load(f"myRes_fold_0.ckpt"))
resmodel0.eval()

resmodel1 = Residual_Network().to(device)
resmodel1.load_state_dict(torch.load(f"myRes_fold_1.ckpt"))
resmodel1.eval()

resmodel2 = Residual_Network().to(device)
resmodel2.load_state_dict(torch.load(f"myRes_fold_2.ckpt"))
resmodel2.eval()

resmodel3 = Residual_Network().to(device)
resmodel3.load_state_dict(torch.load(f"myRes_fold_3.ckpt"))
resmodel3.eval()

resmodel4 = Residual_Network().to(device)
resmodel4.load_state_dict(torch.load(f"myRes_fold_4.ckpt"))
resmodel4.eval()

res18_0 = models.resnet18(pretrained=False)
num_ftrs = res18_0.fc.in_features
res18_0.fc = nn.Linear(num_ftrs, 11)
res18_0 = res18_0.to(device)
res18_0.load_state_dict(torch.load(f"resnet18_fold_0.ckpt"))
res18_0.eval()

res18_1 = models.resnet18(pretrained=False)
num_ftrs = res18_1.fc.in_features
res18_1.fc = nn.Linear(num_ftrs, 11)
res18_1 = res18_1.to(device)
res18_1.load_state_dict(torch.load(f"resnet18_fold_1.ckpt"))
res18_1.eval()

res18_2 = models.resnet18(pretrained=False)
num_ftrs = res18_2.fc.in_features
res18_2.fc = nn.Linear(num_ftrs, 11)
res18_2 = res18_2.to(device)
res18_2.load_state_dict(torch.load(f"resnet18_fold_2.ckpt"))
res18_2.eval()

res18_3 = models.resnet18(pretrained=False)
num_ftrs = res18_3.fc.in_features
res18_3.fc = nn.Linear(num_ftrs, 11)
res18_3 = res18_3.to(device)
res18_3.load_state_dict(torch.load(f"resnet18_fold_3.ckpt"))
res18_3.eval()

res18_4 = models.resnet18(pretrained=False)
num_ftrs = res18_4.fc.in_features
res18_4.fc = nn.Linear(num_ftrs, 11)
res18_4 = res18_4.to(device)
res18_4.load_state_dict(torch.load(f"resnet18_fold_4.ckpt"))
res18_4.eval()

prediction = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        test0 = cnnmodel0(data)
        test0 += cnnmodel1(data)
        test0 += cnnmodel2(data)
        test0 += cnnmodel3(data)
        test0 += cnnmodel4(data)

        test1 = resmodel0(data)
        test1 += resmodel1(data)
        test1 += resmodel2(data)
        test1 += resmodel3(data)
        test1 += resmodel4(data)

        test2 = res18_0(data)
        test2 += res18_1(data)
        test2 += res18_2(data)
        test2 += res18_3(data)
        test2 += res18_4(data)

        test_pred = (test0 + test1 + test2) / 15
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_set)+1)]
df["Category"] = prediction
df.to_csv("ensemble-all.csv",index = False)
print('Finish preficting!')