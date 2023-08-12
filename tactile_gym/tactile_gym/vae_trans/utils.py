import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
import numpy as np
from math import exp
import h5py
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
class TactilePairDataset(Dataset):
    def __init__(self,file_dir, tag, transform):
        self.train_data = []
        self.transform = transform
        self.files = [f for f in os.listdir(file_dir) if tag in f]
        print(self.files)
        self.length = 0
        for file in self.files:
            h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), file_dir,file,'data_vae_pair.h5'), "r")
            self.length = len(h5f_data)
            np_data = np.zeros((len(h5f_data), 128, 128), dtype=np.uint8)
            data = []
            for i in tqdm(range(self.length)):
                dset = h5f_data["data_"+str(i)]
                np_data[i] = dset[:]
            np_data = np_data  / 255   
            print(np_data.shape)
            self.train_data.append(np_data)  
            
    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.stack([self.transform(d[idx]) for d in self.train_data])  

class TactileDataset(Dataset):
    def __init__(self,file_dir, tag, transform):
        self.train_data = []
        self.transform = transform
        self.files = [f for f in os.listdir(file_dir) if tag in f]
        print(self.files)
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
            self.train_data = np_data 
            
    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):

        return torch.tensor(self.train_data[idx])  
 

 
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
    imgs = transform(imgs) * prob_vector + imgs * (1 - prob_vector) # 
    return imgs
    
def random_flip(imgs):
    prob_vector = random_prob(imgs.size(0)).cuda() # 获取每张图像是否要执行形态学操作的标志向量
    prob_vector = prob_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 扩展维度以适应广播机制

    transform = transforms.RandomHorizontalFlip()
    imgs = transform(imgs) * prob_vector + imgs * (1 - prob_vector) # 
    return imgs

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
 
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
 
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
 
    C1 = 0.01**2
    C2 = 0.03**2
 
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
 
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
 
def ssim(img1, img2, window_size = 11, size_average = True):
    img1 = img1*0.5+0.5
    img2 = img2*0.5+0.5
 
    if len(img1.size()) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.size()) == 3:
        img2 = img2.unsqueeze(0)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)



# def add_salt_and_pepper_noise(image_tensor, amount):
#     noisy_image = image_tensor.clone()
#     num_pixels = int(amount * image_tensor.numel())
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Generate random pixel indices for salt noise
#     salt_coords = [torch.randint(0, dim , (num_pixels,)).to(device) for dim in image_tensor.shape]
#     noisy_image[salt_coords[0], salt_coords[1]] = 1.0
    
#     # Generate random pixel indices for pepper noise
#     pepper_coords = [torch.randint(0, dim , (num_pixels,)).to(device) for dim in image_tensor.shape]
#     noisy_image[pepper_coords[0], pepper_coords[1]] = 0.0
    
#     return noisy_image

# def salt_and_pepper(images,prob):
#     # 定义结构元素]
#     thred_l = 0.4
#     thred_h = 0.6

#     noise_tensor= torch.rand_like(images)
#     salt=torch.max(images)
#     pepper=torch.min(images)

#     prob_vector = random_prob(imgs.size(0)).cuda() # 获取每张图像是否要执行形态学操作的标志向量
#     prob_vector = prob_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 扩展维度以适应广播机制

#     transform = transforms.RandomHorizontalFlip()
#     imgs = transform(imgs) * prob_vector + imgs * (1 - prob_vector) # 
#     return imgs

#     images[noise_tensor>1-prob]=pepper    

#     return images


def salt_and_pepper(images, prob):
    """
    Add salt and pepper noise to a batch of images.

    Args:
        images (torch.Tensor): Batch of input images with shape (batch_size, channels, height, width).
        prob (float): Probability of adding salt and pepper noise to each pixel.

    Returns:
        noisy_images (torch.Tensor): Batch of images with added salt and pepper noise.
    """
    noisy_images = images.clone()
    batch_size, channels, height, width = noisy_images.shape

    # Generate random noise tensor with the same shape as images
    noise_tensor = torch.rand(batch_size, channels, height, width, dtype=noisy_images.dtype, device=noisy_images.device)

    # Generate masks for salt and pepper noise
    salt_mask = (noise_tensor < prob / 2)
    pepper_mask = (noise_tensor > 1 - prob / 2)

    # Set salt noise to maximum pixel value and pepper noise to minimum pixel value
    noisy_images[salt_mask] = noisy_images.max()
    noisy_images[pepper_mask] = noisy_images.min()

    return noisy_images