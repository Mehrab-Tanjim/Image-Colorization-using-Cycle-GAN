#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import cv2

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, Dense
import tensorflow as tf


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# modelv2 = InceptionResNetV2( input_shape = (224, 224, 3), weights = "./input/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5")

# import mumpy
images_gray = np.load('./input/l/gray_scale.npy')
images_lab = np.load('./input/ab/ab/ab1.npy')
print(images_gray.shape)
print(images_lab.shape)




# In[3]:


# imgs = np.zeros((1, 224, 224, 3))
# for i in range(0, 3):
#     imgs[0, :, :,i] = images_gray[1029]
# temp_img = preprocess_input(imgs)

# prediction = model_simple.predict(temp_img)

# plt.imshow(prediction[0,:,:,])


# In[4]:


def get_rbg_from_lab(gray_imgs, ab_imgs, n = 10):
    imgs = np.zeros((n, 224, 224, 3))
    imgs[:, :, :, 0] = gray_imgs[0:n:]
    imgs[:, :, :, 1:] = ab_imgs[0:n:]
    print(ab_imgs[0:n:].shape)
    imgs = imgs.astype("uint8")
    
    imgs_ = []
    for i in range(0, n):
        imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

    imgs_ = np.array(imgs_)

#     print(imgs_.shape)
    
    return imgs_
    
temp = get_rbg_from_lab(gray_imgs = images_gray, ab_imgs = images_lab, n = 1)
    
# new_model = Model(inputs = modelv2.inputs, outputs = modelv2.output)


# for i, layer in enumerate(new_model.layers):
#     layer.trainable = False

# x = Reshape((5, 5, 40))(new_model.output)

# x = Conv2DTranspose(strides = 2, kernel_size = 5, filters = 40, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu)(x)
# x = Conv2DTranspose(strides = 3, kernel_size = 7, filters = 40, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu)(x)
# x = Conv2DTranspose(strides = 3, kernel_size = 9, filters = 20, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu)(x)
# x = Conv2DTranspose(strides = 4, kernel_size = 11, filters = 20, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu)(x)

# x = Conv2D(strides = 2, kernel_size = 5, filters = 12, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu)(x)
# x = Conv2D(strides = 1, kernel_size = 9, filters = 3, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu)(x)


# final_model = Model(inputs = new_model.inputs, outputs = x)

#final_model.predict(get_rbg_from_lab(images_gray, images_lab, n = 2)).shape

# final_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), loss = tf.losses.mean_pairwise_squared_error)


# In[5]:


def pipe_line_img(gray_scale_imgs, batch_size = 100, preprocess_f = preprocess_input):
    imgs = np.zeros((batch_size, 224, 224, 3))
    for i in range(0, 3):
        imgs[:batch_size, :, :,i] = gray_scale_imgs[:batch_size]
    return preprocess_f(imgs)

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./folder_to_save_graph_3', histogram_freq=0, write_graph=True, write_images=True)

imgs_for_input = pipe_line_img(images_gray, batch_size = 1500)

imgs_for_output = preprocess_input(get_rbg_from_lab(gray_imgs = images_gray, ab_imgs = images_lab, n = 1500))


# plt.imshow(imgs_for_input)
# print(imgs_for_input.shape)



# plt.imshow(imgs_for_output[1])
# plt.imshow(imgs_for_input[1])

# print(imgs_for_output)


# In[6]:





# model.add(layers.Dense(output_dim=n, activation=softMaxAxis(1)))


# In[7]:


model_simple = Sequential()

model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 16, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu, input_shape = (224, 224, 3)))
model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 32, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu))

model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 64, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu, input_shape = (224, 224, 3)))
# model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 128, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu))
# model_simple.add(Dense(units=1, activation='softmax'))

model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 3, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu))

# model_simple.add(Conv2DTranspose(strides = 1, kernel_size = 3, filters = 3, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu))

# model_simple.add(Conv2D(strides = 1, kernel_size = 3, filters = 12, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu))

# model_simple.add(Conv2DTranspose(strides = 1, kernel_size = 3, filters = 12, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu))
# model_simple.add(Conv2DTranspose(strides = 1, kernel_size = 3, filters = 3, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "same", activation = tf.nn.relu))
# model_simple.add(Conv2DTranspose(strides = 1, kernel_size = 3, filters = 3, use_bias = True, bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) , padding = "valid", activation = tf.nn.relu))

model_simple.compile(optimizer = tf.keras.optimizers.Adam(epsilon = 1e-4), loss = tf.losses.mean_pairwise_squared_error)

# imgs_for_s = np.zeros((1000, 224, 224, 1))
# imgs_for_s[:, :, :, 0] = images_gray[:1000] 

# prediction = model_simple.predict(imgs_for_input)



# In[8]:


model_simple.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


model_simple.fit(imgs_for_input[:1000], imgs_for_output[:1000], epochs = 50, callbacks = [tbCallBack], validation_split = 0.1, shuffle = True)

# model_simple.fit(imgs_for_input, imgs_for_output, epochs = 1100, batch_size = 16)

# prediction = model_simple.predict(imgs_for_input[:1000])


# In[10]:


imgs = np.zeros((1, 224, 224, 3))
for i in range(0, 3):
    imgs[0, :, :,i] = images_gray[1029]
temp_img = preprocess_input(imgs)

prediction = model_simple.predict(temp_img)

image_1029 = np.concatenate((imgs_for_input[1029], imgs_for_output[1029], prediction[0,:,:,]), 1)

# print(img_list.shape)
plt.imshow(image_1029)
plt.axis("off")


# In[13]:


imgs = np.zeros((1, 224, 224, 3))
for i in range(0, 3):
    imgs[0, :, :,i] = images_gray[1029]
temp_img = preprocess_input(imgs)

prediction1 = model_simple.predict(temp_img)

imgs = np.zeros((1, 224, 224, 3))
for i in range(0, 3):
    imgs[0, :, :,i] = images_gray[1028]
temp_img = preprocess_input(imgs)

prediction0 = model_simple.predict(temp_img)

image_0 = np.concatenate((imgs_for_input[1028], imgs_for_output[1028], prediction0[0,:,:,]), 1)
image_1 = np.concatenate((imgs_for_input[1029], imgs_for_output[1029], prediction1[0,:,:,]), 1)

image_new_two = np.concatenate((image_0, image_1), 0)

# print(img_list.shape)
plt.imshow(image_1029)
plt.axis("off")

# plt.imshow(prediction[0,:,:,])


# In[ ]:



    
image_cake = np.concatenate((imgs_for_input[2], imgs_for_output[2], prediction[2,:,:,]), 1)

# print(img_list.shape)
plt.imshow(image_cake)
plt.axis("off")


# In[ ]:


# img_list = []
# img_list.append(
    
image_noidea = np.concatenate((imgs_for_input[5], imgs_for_output[5], prediction[5,:,:,]), 1)

# img_list = np.array(img_list)
# img_list = np.squeeze(img_list, axis=0)
# print(img_list.shape)
plt.imshow(image_noidea)
plt.axis("off")



# In[ ]:


image_two = np.concatenate((image_cake, image_noidea), 0)
plt.imshow(image_two)
plt.axis("off")
                           


# In[ ]:


# #split train and validation data
# import numpy as np
# from matplotlib import pyplot as plt

# images_gray = np.load(opt.dataroot+'A/gray_scale.npy')
# images_lab = np.load(opt.dataroot+'B/ab1.npy')
# np.save(opt.dataroot + 'Train/A/gray_scale.npy', images_gray[:300])
# np.save(opt.dataroot + 'Train/B/ab1.npy', images_lab[:300] )
# images_gray = np.load(opt.dataroot + 'Train/A/gray_scale.npy')
# images_lab = np.load(opt.dataroot + 'Train/B/ab1.npy')
# plt.figure()
# plt.imshow(images_gray[29],cmap='gray')
# plt.show()

# images_gray = np.load(opt.dataroot+'A/gray_scale.npy')
# images_lab = np.load(opt.dataroot+'B/ab1.npy')
# np.save(opt.dataroot + 'Test/A/gray_scale.npy', images_gray[1000:1030])
# np.save(opt.dataroot + 'Test/B/ab1.npy', images_lab[1000:1030])
# images_gray = np.load(opt.dataroot + 'Test/A/gray_scale.npy')
# images_lab = np.load(opt.dataroot + 'Test/B/ab1.npy')
# plt.figure()
# plt.imshow(images_gray[0],cmap='gray')
# plt.show()


# In[ ]:


# trainDataloader = DataLoader(ImageDataset(opt.dataroot + 'Train/', transforms_ = transforms_), batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
# testDataloader = DataLoader(ImageDataset(opt.dataroot + 'Test/', transforms_ = transforms_), batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


