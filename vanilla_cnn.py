#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch.optim as optim

from base_line_cnn_model import *
from base_line_cnn_model import EDcoder_cnn

use_cuda = torch.cuda.is_available()

num_epochs = 30           # Number of full passes through the dataset
batch_size = 15        # Number of samples in each minibatch
learning_rate = 0.0005  
seed = np.random.seed(1) # Seed the random number generator for reproducibility


# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

model = EDcoder_cnn()
model = model.to(computing_device)

criterion = torch.nn.MSELoss(reduce=True, size_average=True)#
optimizer = optim.Adam(model.parameters(),lr=learning_rate) 


# In[2]:


import os
img_path = os.getcwd() + '/img_vanilla_cnn/'

def vis_result(data, target, output, epoch):
    '''visualize images for GAN'''
    #global img_list
    #global pred_rgb
    img_list = []
    
    for i in range(min(32, batch_size)):
    #for i in range(min(2, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0).cpu().numpy()
        raw = target[i].cpu().numpy()
        pred = output[i].cpu().numpy()
        
        raw_rgb = (np.transpose(raw, (1,2,0)).astype(np.float64) )# * 255

        pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64))# * 255
        
        grey = np.transpose(l, (1,2,0))

        grey = (grey.astype(np.float64)) #* 255
#         print(grey.shape)
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        
        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4*i:4*(i+1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)
    #print(np.array(img_list).shape())

    plt.figure(figsize=(18,14))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path + 'epoch%d_val.png' % epoch)
#     plt.clf()
#     plt.show()
    


# In[3]:


def validate(val_loader):
    sum_loss = 0.0
    list_sum_loss = []
    num = 0
    for mb_count, (val_images, val_labels) in enumerate(val_loader):
#         print(mb_count)
        model.eval()
        with torch.no_grad():  

            optimizer.zero_grad()      
            val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
        

            outputs = model(val_images)
            loss = criterion(outputs,val_labels)

            sum_loss += loss

            list_sum_loss.append(sum_loss)
            sum_loss = 0.0
        if mb_count == 0:
            vis_result(val_images.data, val_labels.data, outputs.data, epoch)      
    return sum(list_sum_loss)/(len(list_sum_loss))    


# In[ ]:





# In[4]:


from gan_model import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import time
import os
import sys
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


from load_data_new_ip import ImageNet_Dataset as myDataset
#image_transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()])
# image_transform = transforms.Compose([transforms.ToTensor()])
data_root = ''
data_train = myDataset(data_root, mode='train',
#                       transform=image_transform,
                      types='raw',
                      shuffle=True,
                      large=False
                      )
train_loader = data.DataLoader(data_train,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=4)
data_val = myDataset(data_root, mode='test',
#                       transfortrm=image_transform,
                      types='raw',
                      shuffle=True,
                      large=False
                      )

val_loader = data.DataLoader(data_val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)


# In[6]:


total_loss = []
avg_minibatch_loss = []
total_vali_loss = []
# ones_vec = torch.ones((14, 1))
# threshold_none_diease = 0.5
# tolerence = 6

for epoch in range(num_epochs):
    
    N = 6
    N_minibatch_loss = 0.0    
    early_stop = 0
    # Get the next minibatch of images, labels for training
    i = 0
    for minibatch_count, (images, labels) in enumerate(train_loader):
        
#         print('minibatch_count', minibatch_count)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
    
#         print(images.size())
        model.train()        
        images, labels = images.to(computing_device), labels.to(computing_device)
#         images = torch.reshape(images, (2, 2))

#         labels = labels.long() 
    
        # Zero out the stored gradient (buffer) from the previous iteration
          
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)

#         print(outputs)
#         loss = 0
#         for i in range(len(labels)):
            
        loss = criterion(outputs, labels)#torch.max(labels, 1)[1].long())
        
        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        
#         outputs = torch.sigmoid(outputs)
        
        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss
        
        
        #TODO: Implement validation
        
        if minibatch_count % N == 0 and minibatch_count != 0:    
            
#             output_np = model.get_result().cpu().detach().numpy()
#             label_np = labels.cpu().detach().numpy()
#             print(labels)
#             print(outputs)
            
#             accuracy, precision, recall, bcr = prediction(label_np, output_np)
#             print('accuracy, precision, recall', accuracy, precision, recall)
#             # Print the loss averaged over the last N mini-batches    
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss))
            
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0
            
            model.eval()
            vali_loss = validate(val_loader)
            total_vali_loss.append(vali_loss.item())
            print('validate error',vali_loss.item())
            
#             if total_vali_loss[i] > total_vali_loss[i-1] and i != 0:
#                 early_stop += 1
#                 if early_stop == tolerence:
                    
#                     avg_minibatch_loss = np.array(avg_minibatch_loss)
#                     np.save('avg_minibatch_loss', avg_minibatch_loss)
                    
                    
#                     total_vali_loss = np.array(total_vali_loss)
#                     np.save('total_vali_loss', total_vali_loss)                    
                    
#                     print('early stop here')
#                     break
            
#             i += 1
            
#         if i == 2:
# #             print('i am here')
#             break

    avg_minibatch_loss_np = np.array(avg_minibatch_loss)
    np.save('avg_minibatch_loss', avg_minibatch_loss_np)


    total_vali_loss_np = np.array(total_vali_loss)
    np.save('total_vali_loss', total_vali_loss_np)      
    
    print("Finished", epoch + 1, "epochs of training")
print("Training complete after", epoch + 1, "epochs")


# In[7]:



N = 10
temp_fig = plt.figure(1)
fig = temp_fig.add_subplot(111)
fig.set_xlabel('minibatch_counts', fontsize = 16)
fig.set_ylabel('Loss', fontsize = 16)

epochs = []

for i in range(len(avg_minibatch_loss)):
    epochs.append(N*i)

# epochs = list(N)
# print(epochs)
fig.plot(epochs, avg_minibatch_loss, 'r', linewidth = 4, label = 'Training Set Loss')
fig.plot(epochs, total_vali_loss, 'b', linewidth = 4, label = 'Validation Set Loss')
ttl = 'Loss for training set and validation set'
plt.title(ttl)
fig.legend()
plt.show()


# In[ ]:




