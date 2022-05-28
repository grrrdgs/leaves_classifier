# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
!pip install ttach
!pip install git+https://github.com/ildoonet/cutmix
!pip install resnest --pre
!pip install d2l

# 导入各种包
import torch
import torch.nn as nn
from torch.nn import functional as F
import ttach as tta
from resnest.torch import resnest50

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import d2l.torch as d2l

class TrainVaidDataset(Dataset):
  def __init__(self,data,transform,file_path):
    self.transform=transform
    self.image_name_arr=data['image']
    self.len=data.shape[0]
    self.file_path=file_path
    self.labels_arr=data['label']
  def __getitem__(self,index):
    image_path=self.file_path+self.image_name_arr[index]
    single_img=Image.open(image_path)
    label=self.labels_arr[index]
    num_label=class_to_num[label]
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    return transform(single_img),num_label
  def __len__(self):
    return self.len

class TestDataset(Dataset):
  def __init__(self,data,transform,file_path):
    self.transform=transform
    self.image_name_arr=data['image']
    self.len=data.shape[0]
    self.file_path=file_path
  def __getitem__(self,index):
    image_path=self.file_path+self.image_name_arr[index]
    single_img=Image.open(image_path)
    return self.transform(single_img)
  def __len__(self):
    return self.len

#创建模型
#是否要冻结前面的一些层
def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    model=model
    for param in model.parameters():
      param.requires_grad=False
#Resnest模型
def resnest_model(num_classes,feature_extracting):
  net=resnest50(pretrained=True)
  set_parameter_requires_grad(net,feature_extracting)
  net.fc=nn.Linear(net.fc.in_features,176,bias=True)
  nn.init.xavier_uniform_(net.fc.weight)
  return net

#使用gpu
def get_device():
  if torch.cuda.is_available():
    return torch.device('cuda:0')

#训练
def train():
  for i,(train_ids,vaid_ids) in enumerate(kfold.split(train_data)):
    print(f'第{i+1}则')
    train_subsampler=torch.utils.data.SubsetRandomSampler(train_ids)
    vaid_subsampler=torch.utils.data.SubsetRandomSampler(vaid_ids)
    train_iter=DataLoader(CutMix(TrainVaidDataset(train_data,train_transform,file_path),176,prob=0.5,num_mix=2),32,sampler=train_subsampler,num_workers=4)
    vaid_iter=DataLoader(CutMix(TrainVaidDataset(train_data,train_transform,file_path),176,prob=0.5,num_mix=2),32,sampler=vaid_subsampler,num_workers=4)
    #初始化网络
    model=resnest_model(176,False)
    model=model.to(device)
    #定义优化器
    param_1x=[weight for name,weight in model.named_parameters() if name not in ['fc.weight','fc.bias']]
    updater=torch.optim.AdamW([{'params':param_1x},{'params':model.fc.parameters(),'lr':learning_rate*10}],lr=learning_rate,weight_decay=weight_decay)
    #设置退火学习率
    scheduler=CosineAnnealingLR(updater,10)
    #定义累加器
    metric=d2l.Accumulator(3)
    for epoch in range(epochs):
      model.train()
      print(f'epoch={epoch+1}')
      train_losses,train_acces=[],[]
      for batch in tqdm(train_iter):
        X,y=batch
        X=X.to(device)
        y=y.to(device)
        updater.zero_grad()
        y_hat=model(X)
        train_loss=train_loss_function(y_hat,y)
        train_loss.backward()
        updater.step()
        train_losses.append(train_loss.item())
        train_acc=(y_hat.argmax(dim=-1)==(y.argmax(dim=-1))).float().mean()
        train_acces.append(train_acc)
      print(f'train_loss={sum(train_losses)/len(train_losses):.3f} train_acc={sum(train_acces)/len(train_acces)}')
      #print(f'第{epoch+1}的学习率为{updater.param_groups[0]['lr']}')
      #学习率更新
      scheduler.step()
    #保存model
    save_path=f'fold{i+1}.pth'
    torch.save(model.state_dict(),save_path)
    #模型评估
    model.eval()
    #val_acc=d2l.evaluate_accuracy_gpu(model,vaid_iter)
    vaid_acces=[]
    with torch.no_grad():
      for (X,y) in tqdm(vaid_iter):
        y_hat=model(X.to(device))
        vaid_acc=(y_hat.argmax(dim=-1)==(y.to(device).argmax(dim=-1))).float().mean()
        vaid_acces.append(vaid_acc)
    print(f'val_acc={sum(vaid_acces)/len(vaid_acces)}')

#加载训练好的模型
def get_pred_model(i):
  pred_model=resnest_model(176,False)
  pred_model.to(device)
  path=f'/content/fold{i}.pth'
  pred_model.load_state_dict(torch.load(path))
  return pred_model

def get_result(model_index):
  num_list=[]
  pred_model=get_pred_model(model_index)
  with torch.no_grad():
    for X in tqdm(test_iter):
      y_hat=pred_model(X.to(device))
      num_list.append(y_hat.argmax(dim=-1))
  num_1=torch.stack(num_list).reshape(-1)
  return num_1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载数据集
    train_data = pd.read_csv('/content/drive/MyDrive/dataset/leaves/train.csv')
    test_data = pd.read_csv('/content/drive/MyDrive/dataset/leaves/test.csv')

    file_path = '/content/drive/MyDrive/dataset/leaves/'

    n_train = train_data.shape[0]
    class_to_num = dict(zip(sorted(set(train_data['label'])), range(n_train)))

    num_to_class = {v: k for k, v in class_to_num.items()}

    device=get_device()

    # 定义损失函数
    train_loss_function = CutMixCrossEntropyLoss(True)
    vaid_loss_function = nn.CrossEntropyLoss()

    # 设置随机种子
    torch.manual_seed(0)

    # 定义K则
    K = 5
    kfold = KFold(K, shuffle=False)

    #定义超参数
    epochs = 20
    learning_rate = 5e-5
    weight_decay = 1e-4
    device = get_device()
    batch_size = 32

    #训练
    train()

    #预测
    # 创建数据集
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_iter = DataLoader(TestDataset(test_data, test_transform, file_path), 32, num_workers=4)
    # 加载训练好的模型
    pred_model = resnest_model(176, False)
    pred_model.to(device)
    pred_model.load_state_dict(torch.load('/content/fold1.pth'))

    result1 = get_result(1)
    result2 = get_result(2)
    result3 = get_result(3)
    result4 = get_result(4)
    result5 = get_result(5)
    #合并所有的结果
    all_result = torch.cat((result1, result2, result3, result4, result5)).reshape(5, 8800)
    #投票
    num_result = torch.mode(all_result, dim=0).values
    labels = []
    for i in num_result:
        labels.append(num_to_class[i.item()])
    #保存结果
    test_result = pd.read_csv("/content/drive/MyDrive/dataset/leaves/test.csv")
    test_result['label'] = labels
    test_result.to_csv('result1.csv', index=False)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
