#!/usr/bin/env python
# coding: utf-8

# # **Homework 2 Phoneme Classification**
# 
# * Slides: https://docs.google.com/presentation/d/1v6HkBWiJb8WNDcJ9_-2kwVstxUWml87b9CnA16Gdoio/edit?usp=sharing
# * Kaggle: https://www.kaggle.com/c/ml2022spring-hw2
# * Video: TBA
# * (1%) Simple baseline: 0.45797 (sample code)
# * (1%) Medium baseline: 0.69747 (concat n frames, add layers)
# * (1%) Strong baseline: 0.75028 (concat n, batchnorm, dropout, add layers)
# * (1%) Boss baseline: 0.82324 (sequence-labeling(using RNN))
# 

# In[1]:


# Main link
# !wget -O libriphone.zip "https://github.com/xraychen/shiny-robot/releases/download/v1.0/libriphone.zip"

# Backup Link 0
# !pip install --upgrade gdown
# !gdown --id '1o6Ag-G3qItSmYhTheX6DYiuyNzWyHyTc' --output libriphone.zip

# Backup link 1
# !pip install --upgrade gdown
# !gdown --id '1R1uQYi4QpX0tBfUWt2mbZcncdBsJkxeW' --output libriphone.zip

# Backup link 2
# !wget -O libriphone.zip "https://www.dropbox.com/s/wqww8c5dbrl2ka9/libriphone.zip?dl=1"

# Backup link 3
# !wget -O libriphone.zip "https://www.dropbox.com/s/p2ljbtb2bam13in/libriphone.zip?dl=1"

# !unzip -q libriphone.zip
# !ls libriphone

# In[3]:


import os
import random
# from tqdm import tqdm
import numpy as np

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    # return x.permute(1, 0, 2) # seq_len, concat_n, feature_dim
    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    # print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    X = []
    # max_len = 3000000
    # X = torch.empty(max_len, concat_nframes, 39)
    # X = torch.empty(len(usage_list), 1000, concat_nframes * 39)
    if mode != 'test':
        y = []
        # y = torch.empty(max_len, dtype=torch.long)
        # y = torch.empty(len(usage_list), 1000, dtype=float)
    min_seq_len = 1000
    max_seq_len = 0
    # idx = 0
    for i, fname in enumerate(usage_list):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        min_seq_len = min(min_seq_len, cur_len)
        max_seq_len = max(max_seq_len, cur_len)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])

        # X[idx, :cur_len, :] = feat
        X += [feat]       
        if mode != 'test':
            y += [label]
            # print(label.shape)
            # y[idx, :cur_len] = label

        # idx += cur_len
    print(f'''[Dataset] # phone classes: {class_num} # number of utterances for {split}: {len(usage_list)}''')
    # print('max seq lens:{max_seq_len}')

    # X = X[:idx]
    # if mode != 'test':
    #     y = y[:idx]

    print(f'[INFO] {split} set')
    print(f"X.shape: {len(X)}, seq lens: {min_seq_len}~{max_seq_len}, {39*concat_nframes}")
    if mode != 'test':
        print(f"y.shape: {len(y)}, seq lens: {min_seq_len}~{max_seq_len}")
        return X, y
    else:
        return X


# In[4]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label == None:
            return self.data[idx], None

        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# In[5]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# data prarameters
concat_nframes = 1              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 850221                        # random seed
batch_size = 64      # 400          # batch size
seq_len = concat_nframes             # RNN seq lens
num_epoch = 100                   # the number of training epoch
learning_rate = 0.001          # learning rate
dropout = 0.5
# model_path = './model.ckpt'     # the path where the checkpoint will be saved
model_name = 'BiLSTM_1concat' # Linear or ResNet
# model parameters
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 3                # the number of hidden layers
hidden_dim = 256       # the hidden dim


# In[6]:


import gc
import torch.nn.utils.rnn as rnn_utils
# pad variable seq_lens
def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    seq_lens = [len(d[0]) for d in data]
    train = rnn_utils.pad_sequence([t for t, _ in data], batch_first=True)
    label = rnn_utils.pad_sequence([l for _, l in data], batch_first=True)
    return train, label, seq_lens


# preprocess data
phone_path = './libriphone'
feat_dir = './libriphone/feat'
train_X, train_y = preprocess_data(split='train', feat_dir = feat_dir, phone_path = phone_path, concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir=feat_dir, phone_path = phone_path, concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader  
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)


# In[7]:


# batch_x, batch_y, seq_lens = iter(train_loader).next()
# print(batch_x.shape, batch_y.shape, len(seq_lens))

# output:
# torch.Size([512, 998, 39]) torch.Size([512, 998]) 512


# In[8]:


class myLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers = 4, batch_first = True, bidirectional = True, dropout = 0.5):
        super(myLSTM, self).__init__()

        self.lstm = nn.RNN(input_dim, hidden_dim, num_layers = num_layers, batch_first = batch_first, bidirectional = bidirectional, dropout = 0)

        self.linear = nn.Linear(hidden_dim * 2, 41)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, batch_first = batch_first, bidirectional = bidirectional, dropout = dropout, proj_size = 41)
        # self.linear = nn.Linear(82, 41)

  def forward(self, x):
    #   (h, c)
      x, h = self.lstm(x)
      # x, lens = rnn_utils.pad_packed_sequence(x, batch_first=True)
      x1 = self.linear(x[0])
      return x1, x[1]


# In[9]:


# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
model = myLSTM(input_dim, hidden_dim, num_layers = hidden_layers, batch_first = True, bidirectional = True, dropout = dropout).to(device)
# model = nn.LSTM(39, hidden_dim, num_layers = hidden_layers, batch_first = True, bidirectional = True, dropout = dropout).to(device)
# model.load_state_dict(torch.load(f"./{model_name}.ckpt"))
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
print(model)


# In[10]:


best_acc = 0.0

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    train_cnt = 0
    loss_cnt = 0
    # training
    model.train() # set the model to training mode
    for i, [features, labels, seq_list] in enumerate(train_loader):
        # features.shape = [batch.size, seq_len, feature_len]
        features = rnn_utils.pack_padded_sequence(features, lengths=seq_list, batch_first=True)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, lens = model(features)
        outputs, _ = rnn_utils.pad_packed_sequence(rnn_utils.PackedSequence(outputs, batch_sizes=lens.cpu()), batch_first=True)
        # loss
        loss = 0
        loss_cnt += len(outputs)
        for o, l, seq in zip(outputs, labels, seq_list):
            train_cnt += seq
            val = criterion(o[:seq], l[:seq])
            loss += val
            _, train_pred = torch.max(o[:seq], 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == l[:seq].detach()).sum().item()
            train_loss += val.item()
        loss.backward()
        optimizer.step()
        # scheduler
        # scheduler1.step()
    
    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        val_cnt = 0
        val_loss_cnt = 0
        with torch.no_grad():
            for i, [features, labels, seq_list] in enumerate(val_loader):
                features = rnn_utils.pack_padded_sequence(features, lengths=seq_list, batch_first=True)
                features = features.to(device)
                # labels = rnn_utils.pack_padded_sequence(labels, lengths=seq_list, batch_first=True)
                labels = labels.to(device)
                
                outputs, lens = model(features)
                outputs, _ = rnn_utils.pad_packed_sequence(rnn_utils.PackedSequence(outputs, batch_sizes=lens.cpu()), batch_first=True)
                # outputs = model(features)
                # outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)

                val_loss_cnt += len(outputs)
                for o, l, seq in zip(outputs, labels, seq_list):
                    val_cnt += seq
                    loss = criterion(o[:seq], l[:seq])
                    _, val_pred = torch.max(o[:seq], 1) # get the index of the class with the highest probability
                    val_acc += (val_pred.detach() == l[:seq].detach()).sum().item()
                    val_loss += loss.item()

                # loss = criterion(outputs, labels) 
                # _, val_pred = torch.max(outputs, 1) 
                # val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                # val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/train_cnt, train_loss/loss_cnt, val_acc/val_cnt, val_loss/val_loss_cnt
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"./{model_name}.ckpt")
                print('saving model with acc {:.6f}'.format(best_acc/val_cnt))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/train_cnt, train_loss/loss_cnt
        ))
# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), f"./{model_name}.ckpt")
    print('saving model at last epoch')


# In[ ]:


del train_loader, val_loader
gc.collect()

def collate_fn(data):
    data = [[i, len(t), t]for i, (t, l) in enumerate(data)]
    data.sort(key=lambda x: x[1], reverse=True)

    seq_lens = [d[1] for d in data]
    idx = [d[0] for d in data]
    train = rnn_utils.pad_sequence([d[2] for d in data], batch_first=True)
    
    return train, seq_lens, idx

test_X = preprocess_data(split='test', feat_dir=feat_dir, phone_path=phone_path, concat_nframes=concat_nframes)
test_set = LibriDataset(test_X)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)


# # **record models**
# 
# * first
#     * model = myLSTM(7*39, 1024, num_layers = 3, batch_first = True, bidirectional = True, dropout = 0.5).to(device)
#     * model.load_state_dict(torch.load(f"./BiLSTM.ckpt"))
#     * acc rate: 0.79
# 
# * second
#     * model = myLSTM(7*39, 256, num_layers = 4, batch_first = True, bidirectional = True, dropout = 0.5).to(device)
#     * model.load_state_dict(torch.load(f"./BiLSTM2.ckpt"))
# 

# In[ ]:


'''model 1'''
# model = myLSTM(7*39, 256, num_layers = 4, batch_first = True, bidirectional = True, dropout = 0.5).to(device)
# batch_size = 32
# model.load_state_dict(torch.load(f"./BiLSTM.ckpt"))
# criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
# acc rate: 0.795
'''model 2'''
# model = myLSTM(15*39, 256, num_layers = 4, batch_first = True, bidirectional = True, dropout = 0.5).to(device)
# batch_size = 32
# model.load_state_dict(torch.load(f"./BiLSTM_15concat.ckpt"))
# criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# acc rate: 0.78
'''model 3'''
# batch_size = 16
# model.load_state_dict(torch.load(f"./BiLSTM_1concat.ckpt"))
# model = myLSTM(1*39, 512, num_layers = 4, batch_first = True, bidirectional = True, dropout = 0.5).to(device)
# criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# acc rate: 


# In[ ]:



pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    
    for i, [features, seq_list, idx] in enumerate(test_loader):
        def f(i):
          return idx[i]
        features = rnn_utils.pack_padded_sequence(features, lengths=seq_list, batch_first=True)
        features = features.to(device)
        # labels = rnn_utils.pack_padded_sequence(labels, lengths=seq_list, batch_first=True)
        labels = labels.to(device)
        
        outputs, lens = model(features)
        outputs, _ = rnn_utils.pad_packed_sequence(rnn_utils.PackedSequence(outputs, batch_sizes=lens.cpu()), batch_first=True)
        # outputs = model(features)
        # outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        
        for output, seq, _ in sorted(zip(outputs, seq_list, range(len(outputs))), key = lambda x: f(x[2])):
          _, test_pred = torch.max(output, 1) # get the index of the class with the highest probability
          pred = np.concatenate((pred, test_pred[:seq].cpu().numpy()), axis=0)


with open(f'prediction_15concat.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))


# In[ ]:


len(pred)
# should be equal to 646268

