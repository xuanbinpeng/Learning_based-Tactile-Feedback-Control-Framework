from VAE_cross_domain.model import DisentangledVAE,Discriminator,FeatureMapping
from VAE_cross_domain.utils import *

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tactile = cv2.VideoCapture(2)

Model_R = DisentangledVAE
Model_S = DisentangledVAE
Model_D = Discriminator
Model_M = FeatureMapping

model_R = Model_R().eval()
model_S = Model_S().eval()

model_R.load_state_dict(torch.load("./VAE_cross_domain/checkpoints/0_model.pt"))
model_S.load_state_dict(torch.load("./VAE_cross_domain/checkpoints/1_model.pt"))

model_R = model_R.to(device)
model_S = model_S.to(device)

#-------------------------------------
model_D = Model_D()
model_M = Model_M().eval()

model_D.load_state_dict(torch.load("./VAE_cross_domain/checkpoints/discriminator_model.pt"))
model_M.load_state_dict(torch.load("./VAE_cross_domain/checkpoints/mapping_model.pt"))
#print(model_M)

model_D = model_D.to(device)
model_M = model_M.to(device)
        
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
    frame = torch.tensor(frame, dtype=torch.float32).to(device)

    return frame
    
def get_tactile_vec(frame):

    mu, _ = model_R.encoder(frame)
    mu_sim = model_M(mu)
    fakeimage = model_S.decoder(mu_sim)
        
    fakeimage = fakeimage.cpu().detach().numpy()
    fakeimage = fakeimage.reshape((128, 128))
    fakeimage *= 255

    
        
    cv2.imshow('fakeimage',fakeimage)
    obs_vector = mu.detach().cpu().numpy()[0]
    return list(obs_vector)
 
    
if __name__ == '__main__':
    while(True):
        frame_test = get_tactile_img() 
        vector = get_tactile_vec(frame_test) 
        if cv2.waitKey(5) & 0xFF == ord( 'q' ) :
            break
        #print(vector)
