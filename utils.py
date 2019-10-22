import random
import time
import datetime
import sys
import pickle
from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
from matplotlib import pyplot as plt

from skimage import color

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        #self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_list = {}
        self.epochs_list = [self.epoch-1] #starts with 0
        self.loss_windows = {}
        self.image_windows = {}
        self.mode = 'train'
        self.current_loss = 0
        self.best_loss = 100000
        self.lab = True

    def log(self, losses=None, images=None, mode='train', model = None, lab = True):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()
        self.mode = mode

        if (self.batch % 100) == 0:
            sys.stdout.write('\r'+self.mode+'Epoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name == 'loss_G_identity' and losses[loss_name]==0:
                temp = 0
            else:
                temp = losses[loss_name].data
            if loss_name not in self.losses:
                self.losses[loss_name] = temp
            else:
                self.losses[loss_name] += temp 
            
            if (self.batch % 100) == 0:
                
                if (i+1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        
        if (self.batch % 100) == 0:
            sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))


        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            
            #save for future reference
            with open('output/losses_'+str(self.mode)+'.pickle', 'wb') as handle:
                pickle.dump(self.losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

#             with open('filename.pickle', 'rb') as handle:
#                 b = pickle.load(handle)
            
            for loss_name, loss in self.losses.items():
                if loss_name == 'loss_G_identity' and loss == 0:
                    temp = 0
                else:
                    temp = loss.cpu()/self.batch
                    
                if loss_name not in self.loss_list:
                    self.loss_list[loss_name] = [temp]
                else:
                    self.loss_list[loss_name].append(temp)
                
                self.current_loss += temp
                
                fig = plt.figure()
                
                if self.mode == 'train':
                    plt.plot(self.epochs_list, self.loss_list[loss_name])
                else:
                    plt.plot(self.epochs_list, self.loss_list[loss_name], 'y')
                
                fig.suptitle(loss_name, fontsize=20)
                plt.xlabel('number of epochs', fontsize=14)
                plt.ylabel('loss', fontsize=14)
                #plt.ioff()
                fig.savefig('output/'+loss_name+'_'+str(self.mode)+'.png')
                plt.close()

                # Reset losses for next epoch
                self.losses[loss_name] = 0.0
            
                   
            #draw images
            img_list = []
    
            image_list = []
            for image_name, tensor in images.items():
                
                
                img = tensor.data.cpu().numpy()[0].transpose(1,2,0)
                
                if self.lab==True:
                    
                    img[:,:,0] = img[:,:,0] * 100. # L channel 0-100
                    img[:,:,1] = img[:,:,1] * 110. # ab channel -110 - 110
                    img[:,:,2] = img[:,:,2] * 110. # ab channel -110 - 110


                    img = color.lab2rgb(img)*255
                    img = img.astype(np.uint8)
                    
                image_list.append(img)
    
            
            img_list.append(np.concatenate((image_list), 1))
            img_list = np.array(img_list)
            img_list = np.squeeze(img_list, axis=0)
            
            plt.axis("off")
            plt.imsave('output/image_'+self.mode+'_'+str(self.epoch)+'.png', img_list)
            plt.axis("on")
            
            #save best
            print("current "+self.mode +" total loss", self.current_loss)

            if (self.best_loss> self.current_loss and self.mode=='validation'):
                print("current is the best loss")
                self.best_loss = self.current_loss
                # Save models checkpoints
                torch.save(model['netG_A2B'], 'output/netG_A2B.pth')
                torch.save(model['netG_B2A'], 'output/netG_B2A.pth')
                torch.save(model['netD_A'], 'output/netD_A.pth')
                torch.save(model['netD_B'], 'output/netD_B.pth')
            
            #reset
            
            self.epochs_list.append(self.epoch)
            self.epoch += 1
            self.batch = 1
            self.current_loss = 0
            sys.stdout.write('\n')

        else:
            self.batch += 1            

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

