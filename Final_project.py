# -*- coding: utf-8 -*-
"""Untitled27.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ODCTbgjM6FIZEw7Ix59CUFlhkGMate9g
"""

from google.colab import drive
drive.mount('/content/drive')

!cp '/content/drive/My Drive/traffic-signs-data.zip' '/content/traffic-signs-data.zip'

!unzip /content/traffic-signs-data.zip

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

with open('/content/train.p','rb') as f:
  train = pickle.load(f)

with open('/content/valid.p', 'rb') as f:
  eval = pickle.load(f)

X_train, y_train = train['features'],train['labels']
X_eval , y_eval = eval['features'],eval['labels']

print(X_train.shape)
print(y_train.shape)
print(X_eval.shape)
print(y_eval.shape)

print(y_train[0])
plt.imshow(X_train[0])
plt.show()

n_classes,counts = np.unique(y_train,return_counts=True)
print(n_classes)

plt.rcParams['figure.figsize']=[15,5]
plt.bar(n_classes,counts,tick_label=n_classes,width=0.8,align='center')
plt.xlabel('nclasses')
plt.ylabel('counts')
plt.show()

import torchvision.transforms as transforms
import random
trans = transforms.Compose([transforms.ToTensor()])

def next_batch(X,y,batch_size,shuffle=True):
  start_index = 0
  arr = np.arange(0,X.shape[0])
  if shuffle==True:
      random.shuffle(arr)


  while start_index < X.shape[0]:
    images = X[arr[start_index:start_index+batch_size],:,:,:]
    images = np.transpose(images,(0,3,1,2))
    images = torch.Tensor(images)
    labels = y[arr[start_index:start_index+batch_size]]
    labels = torch.Tensor(labels)
    # print(type(images))
    # print(images.shape)
    
    yield (images,labels)

    start_index+=batch_size

import torch.nn as nn

class LeNet(nn.Module):

  def __init__(self):
    super(LeNet,self).__init__()

    self.cnn_model = nn.Sequential(
        nn.Conv2d(3,6,5),
        nn.Tanh(),
        nn.AvgPool2d(2,stride=2),
        nn.Conv2d(6,16,5),
        nn.Tanh(),
        nn.AvgPool2d(2,stride=2)
    )

    self.fc_model = nn.Sequential(
        nn.Linear(400,120),
        nn.Tanh(),
        nn.Linear(120,84),
        nn.Tanh(),
        nn.Linear(84,43)
    )

  def forward(self,X):
    X = self.cnn_model(X)
    #X = X.view(X.size(0),-1)
    X = X.reshape(X.size(0),-1)
    X = self.fc_model(X)
    return X

net = LeNet().to(device)
display(net)

# u=np.random.rand(16,5,5)
# print(u.shape)
# u = u.reshape(-1)
# print(u.shape)

# def evaluation(X,y,model,batch_size,data_size,data_gen):
#   n_batches = math.ceil(data_size/batch_size)
#   last_batch_size = data_size%batch_size

#   accuracy=[]
  
#   for _ in range(n_batches):
#     images,labels = next(data_gen)
#     output = model(images)
#     _,pred = torch.max(output.data,1)

def calculate_accuracy(data_gen,model,data_size,batch_size):
  tot,corr=0,0
  n_batches = math.ceil(data_size/batch_size)
  last_batch = data_size%batch_size
  for _ in range(n_batches):
    images,labels = next(data_gen)
    images,labels = images.to(device),labels.to(device)
    # images = trans(images)
    # labels = trans(labels)

    output = model(images)
    _,pred = torch.max(output.data,1)
    pred = pred.to(device)
    tot+=labels.size(0)
    corr+=(pred==labels).sum().item()

    del images,labels,pred
  return 100*corr/tot

import torch.optim as optim
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters())

import math
loss_arr=[]
loss_epoch_arr=[]
max_epochs=15
batch_size=64
for epoch in range(max_epochs):
  train_gen = next_batch(X_train,y_train,batch_size,shuffle=True)
  n_batches_train=math.ceil(X_train.shape[0]/batch_size)

  for _ in range(n_batches_train):
    images,labels = next(train_gen)
    images,labels = images.to(device),labels.to(device)
    # images = trans(images)
    # labels = trans(labels)
    opt.zero_grad()
    outputs = net(images)
    outputs = outputs.to(device)  #Dont miss this or else loss will be at 0 all time

    loss = loss_fn(outputs,labels.long())
    loss.backward()
    opt.step()
    loss_arr.append(loss.item())
    del images,labels,outputs

  loss_epoch_arr.append(loss.item())
  with torch.no_grad():  
    train_gen = next_batch(X_train,y_train,batch_size,shuffle=False)
    train_size = X_train.shape[0]
    train_acc = calculate_accuracy(train_gen,net,train_size,batch_size)

    valid_gen = next_batch(X_eval,y_eval,batch_size,shuffle=False)
    valid_size = X_eval.shape[0]
    eval_acc = calculate_accuracy(valid_gen,net,valid_size,batch_size)

  
  print('Epoch: %d/%d , Train acc: %0.2f, Eval acc: %0.2f '%(epoch,max_epochs,train_acc,eval_acc))

plt.plot(loss_epoch_arr)
plt.show()

torch.save(net.state_dict(),'./model.pth')

net = LeNet()
net.load_state_dict(torch.load('./model.pth',map_location=torch.device('cpu')))
net = net.to(device)

with open('/content/test.p','rb') as f:
  test = pickle.load(f)

X_test, y_test = test['features'],test['labels']
print(X_test.shape)
print(y_test.shape)

from sklearn import metrics
import seaborn as sns
def confusion_matrix(data_gen,model,data_size,batch_size):
  tot,corr=0,0
  n_batches = math.ceil(data_size/batch_size)
  last_batch = data_size%batch_size
  y=[]
  y_hat=[]
  for _ in range(n_batches):
    images,labels = next(data_gen)
    images,labels = images.to(device),labels.to(device)
    # images = trans(images)
    # labels = trans(labels)

    output = model(images)
    _,pred = torch.max(output.data,1)
    u = pred.clone().detach()
    y_hat.extend(u)
    y.extend(labels)

    cm = metrics.confusion_matrix(y,y_hat)
    pred = pred.to(device)
    tot+=labels.size(0)
    corr+=(pred==labels).sum().item()

    del images,labels,pred

  return 100*corr/tot,cm

net.eval()
test_size = X_test.shape[0]
test_gen = next_batch(X_test,y_test,test_size,shuffle=False)
print(calculate_accuracy(test_gen,net,test_size,test_size))

net.eval()
test_size = X_test.shape[0]
test_gen = next_batch(X_test,y_test,test_size,shuffle=False)

acc,cm=confusion_matrix(test_gen,net,test_size,test_size)
print(acc)
plt.figure(figsize = (42,42))

ax=sns.heatmap(cm,annot=True)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicated Label')

plt.savefig('confusion_matrix.png')

inp = 5
print(y_test[inp])

im = X_test[inp]
im = np.expand_dims(im,axis=0)
im = np.transpose(im,(0,3,1,2))
im = torch.Tensor(im)
im = im.to(device)
out = net(im)
_,pred = torch.max(out.data,1)
print(pred)
plt.imshow(X_test[inp])
plt.show()

display(net)

v = net.cnn_model[3].weight.data
print(v.shape)

import seaborn as sns
def plot_filters_single_channel_big(t):
  nrows = t.shape[0]*t.shape[2]
  ncols = t.shape[1]*t.shape[3]

  npimg = np.array(t.numpy(),np.float32)
  npimg = npimg.transpose((0,2,1,3))
  print(npimg.shape)
  npimg = npimg.ravel().reshape(nrows,ncols)
  print(npimg.shape)
  npimg = npimg.T
  print(npimg.shape)

  fig,ax = plt.subplots(figsize = (ncols,nrows))
  imgplot = sns.heatmap(npimg,xticklabels=False,yticklabels=False,cmap='Greys',ax=ax,cbar=False)

def plot_filters_single_channel(t):
  nplots = t.shape[0]*t.shape[1]
  ncols = 6
  nrows = 1 + nplots//ncols

  npimg = np.array(t.numpy(),np.float32)
  count=0
  fig = plt.figure(figsize=(ncols,nrows))
  for i in range(t.shape[0]):
    for j in range(t.shape[1]):
      count+=1
      ax1 = fig.add_subplot(nrows,ncols,count)
      npimg = np.array(t[i,j].numpy(),np.float32)
      ax1.imshow(npimg)
      ax1.set_title(str(i) + ',' + str(j))
      ax1.axis('off')
      ax1.set_xticklabels([])
      ax1.set_yticklabels([])
   
  plt.tight_layout()
  plt.show()

def plot_filters_multi_channel(t):
  num_kernels = t.shape[0]
  num_cols = 6
  num_rows = num_kernels

  fig=plt.figure(figsize=(num_cols,num_rows))

  for i in range(t.shape[0]):
    ax1 = fig.add_subplot(num_rows,num_cols,i+1)

    npimg = np.array(t[i].numpy(),np.float32)
    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
    npimg = npimg.transpose((1,2,0))
    ax1.imshow(npimg)
    ax1.axis('off')
    ax1.set_title(str(i))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

  plt.tight_layout()
  plt.show()

def plot_weights(model,layer_num,single_channel=True,collated=False):
  layer = model.cnn_model[layer_num]

  if isinstance(layer,nn.Conv2d):
    weight_tensor = model.cnn_model[layer_num].weight.data
    print(weight_tensor.shape)
    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)
    else:
      if weight_tensor.shape[1]==3:
        plot_filters_multi_channel(weight_tensor)
      else:
        print('Not possible')
  else:
    print('Can visualize only Conv layers')

plot_weights(net,0,True,True)

plot_weights(net,0,True,False)

plot_weights(net,0,False,False)

plot_weights(net,3,True,False)

inp = 5
print(y_test[inp])

im = X_test[inp]
im = np.expand_dims(im,axis=0)
im = np.transpose(im,(0,3,1,2))
im = torch.Tensor(im)
im = im.to(device)
out = net(im)
_,pred = torch.max(out.data,1)
print(pred)
plt.imshow(X_test[inp])
plt.show()