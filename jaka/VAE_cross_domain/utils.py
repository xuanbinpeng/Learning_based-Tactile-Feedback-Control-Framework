import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
import numpy as np
from math import exp
import h5py
import cv2
from tqdm import tqdm

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std


def obs_extract(obs):
    obs = np.transpose(obs['rgb'], (0,3,1,2))
    return torch.from_numpy(obs)


def count_step(i_update, i_env, i_step, num_envs, num_steps):
    step = i_update * (num_steps *  num_envs) + i_env * num_steps + i_step
    return step


# for representation learning
class PairDataset(Dataset):
    def __init__(self, file_dir, tag, transform):
        self.train_data_real = []
        self.train_data_sim = []

        self.transform = transform
        self.files = [f for f in os.listdir(file_dir) if tag in f]
        self.length = len(self.files[0])
        
        for file in os.listdir(file_dir+'/'+self.files[0]):
            #print(os.path.join(os.path.dirname(__file__), file_dir,self.files[0],file))
            img = cv2.imread(os.path.join(os.path.dirname(__file__), file_dir,self.files[0],file),cv2.IMREAD_GRAYSCALE)
            #print(img)
            img = img / 255
            self.train_data_real.append(img) 

        for file in os.listdir(file_dir+'/'+self.files[1]):
            #print(os.path.join(os.path.dirname(__file__), file_dir,self.files[0],file))
            img = cv2.imread(os.path.join(os.path.dirname(__file__), file_dir,self.files[1],file),cv2.IMREAD_GRAYSCALE)
            #print(img)
            img = img / 255
            self.train_data_sim.append(img) 
        self.train_data = [self.train_data_real,self.train_data_sim]
        #print(len(self.train_data))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #for d in self.train_data:
        #    print(len(d))
        return torch.stack([self.transform(d[idx]) for d in self.train_data])

class TactileDataset(Dataset):
    def __init__(self,file_dir, tag, transform):
        self.train_data = []
        self.transform = transform
        self.files = [f for f in os.listdir(file_dir) if tag in f]
        self.length = 0
        for file in self.files:
            h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), file_dir,file,'data_vae_all.h5'), "r")
            self.length = len(h5f_data)
            np_data = np.zeros((len(h5f_data), 128, 128), dtype=np.uint8)
            data = []
            for i in tqdm(range(self.length)):
                dset = h5f_data["data_"+str(i)]
                np_data[i] = dset[:]
            np_data = np_data  / 255   
            print(np_data.shape)
            print(type(np_data))
            self.train_data.append(np_data)  
            print(len(self.train_data))
            
            
    
    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.stack([self.transform(d[idx]) for d in self.train_data])  
 

 
# 定义腐蚀函数
def erode(images, kernel):
    # 对图像取反，使得白色为0，黑色为1
    images = 1 - images 
    # 对图像进行卷积操作，并取最大值
    output = F.conv2d(images, kernel, padding=1)
    output = torch.max(output - (kernel.sum() - 1), torch.zeros_like(output))
    # 对输出图像再次取反，恢复原来的颜色
    output = 1 - output 
    return output
 
# 定义膨胀函数
def dilate(images,kernel):
    # 对图像进行卷积操作，并取最小值
    output = F.conv2d(images, kernel, padding=1)
    output = torch.min(output + (kernel.sum() - 1), torch.ones_like(output))
    return output
 
# 定义随机概率函数
def random_prob(batch_size):
    # 创建一个0-1之间的随机数向量，并根据阈值判断是否为1或0
    threshold = 0.5 # 设置阈值为0.5，即50%的概率执行形态学操作
    prob_vector = torch.rand(batch_size) # 创建一个长度为batch_size的随机数向量
    prob_vector = (prob_vector > threshold).float() # 将大于阈值的位置设为1，小于阈值的位置设为0
    return prob_vector
 
# 应用随机概率函数和形态学操作函数
def random_augmentaction(images):
    # 定义结构元素
    threshold = 0.5
    prob = torch.rand(1)
    kernel = torch.ones(3, 3).cuda() # 一个3x3的方形结构元素
    kernel = kernel.unsqueeze(0).unsqueeze(0) # 扩展维度以适应卷积操作
    prob_vector = random_prob(images.size(0)).cuda() # 获取每张图像是否要执行形态学操作的标志向量
    prob_vector = prob_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 扩展维度以适应广播机制
 
    if prob > threshold:
 
        images = erode(images,kernel) * prob_vector + images * (1 - prob_vector) # 按照一定概率对图像进行腐蚀操作，并保留原始图像中不执行操作的部分
        #images = erode(images,kernel) * prob_vector + images * (1 - prob_vector) # 按照一定概率对图像进行腐蚀操作，并保留原始图像中不执行操作的部分

    else:
        images = dilate(images,kernel) * prob_vector + images * (1 - prob_vector) # 按照一定概率对图像进行膨胀操作，并保留原始图像中不执行操作的部分
        #images = dilate(images,kernel) * prob_vector + images * (1 - prob_vector) # 按照一定概率对图像进行膨胀操作，并保留原始图像中不执行操作的部分

    return images


#random resize
def random_crop(imgs):
    prob_vector = random_prob(imgs.size(0)).cuda() # 获取每张图像是否要执行形态学操作的标志向量
    prob_vector = prob_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 扩展维度以适应广播机制
    size = (128,128)
    scale = (0.85, 1.0)
    ratio = (0.85, 1.0)
    transform = transforms.RandomResizedCrop(size=size, scale=scale,ratio=ratio)
    imgs = transform(imgs) * prob_vector + imgs * (1 - prob_vector) # 按照一定概率对图像进行腐蚀操作，并保留原始图像中不执行操作的部分
    return imgs
    
def random_flip(imgs):
    prob_vector = random_prob(imgs.size(0)).cuda() # 获取每张图像是否要执行形态学操作的标志向量
    prob_vector = prob_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 扩展维度以适应广播机制

    transform = transforms.RandomHorizontalFlip()
    imgs = transform(imgs) * prob_vector + imgs * (1 - prob_vector) # 按照一定概率对图像进行腐蚀操作，并保留原始图像中不执行操作的部分
    return imgs

