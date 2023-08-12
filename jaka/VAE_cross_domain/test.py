from model import DisentangledVAE,Discriminator,FeatureMapping
from utils import *

import cv2
import os
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import torch
from torch import optim
from torch.nn import functional as F
import torch.nn 
from torch.autograd import Variable

Model_R = DisentangledVAE
Model_S = DisentangledVAE
Model_D = Discriminator
Model_M = FeatureMapping

tactile = cv2.VideoCapture(0)

def get_tactile_img():
    ret,frame = tactile.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame = frame[80:480,100:530]
    (h, w) = frame.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 132, 1.0) 
    frame = cv2.warpAffine(frame, M, (w, h))
    frame = cv2.resize(frame,(128,128))
    rett,frame = cv2.threshold(frame,60,255,cv2.THRESH_BINARY)
    #frame = cv2.dilate(frame, cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)),iterations = 1)
    cv2.imshow('Tactile',frame)

    frame = frame / 255
    frame = frame.reshape((1,1,128,128))
    frame = torch.tensor(frame, dtype=torch.float32)

    return frame
    
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------------------------------------
    model_R = Model_R()
    model_S = Model_S()
    file_path = os.path.join(".", "checkpoints", "0_model.pt")
    model_R.load_state_dict(torch.load(file_path))
    model_S.load_state_dict(torch.load("./checkpoints/1_model.pt"))
    # model_R.load_state_dict(torch.load("checkpoints/0_model.pt"),map_location=torch.device('cpu'))
    # model_S.load_state_dict(torch.load("checkpoints/1_model.pt"),map_location=torch.device('cpu'))

    model_R = model_R.to(device)
    model_S = model_S.to(device)

#-------------------------------------
    model_D = Model_D()
    model_M = Model_M().eval()

    model_D.load_state_dict(torch.load("/home/bryant/Tac/jaka/VAE_cross_domain/checkpoints/discriminator_model.pt",map_location=torch.device('cpu')))
    model_M.load_state_dict(torch.load("/home/bryant/Tac/jaka/VAE_cross_domain/checkpoints/mapping_model.pt",map_location=torch.device('cpu')))
    #print(model_M)
    
    model_D = model_D.to(device)
    model_M = model_M.to(device)
    while(True):
        img = get_tactile_img() 

        mu,_ = model_R.encoder(img)
        mu_sim = model_M(mu)
        recon_imgs = model_S.decoder(mu_sim).cpu().detach().numpy()

        recon_imgs = recon_imgs.reshape((128,128))
        recon_imgs = recon_imgs*255
        
        cv2.imshow("recon:",recon_imgs)
        #time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.waitKey(0)
            break



