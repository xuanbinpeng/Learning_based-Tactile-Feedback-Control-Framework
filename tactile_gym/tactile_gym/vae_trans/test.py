from model import VAE,Discriminator,FeatureMapping
from utils import *

import cv2
import os
import numpy as np
import time

import torch
from torch.nn import functional as F
import torch.nn 

Model_R = VAE
Model_S = VAE
Model_D = Discriminator
Model_M = FeatureMapping

real_path = './data/tactile_pair/tactile1/data_vae_pair.h5'
sim_path =  './data/tactile_pair/tactile0/data_vae_pair.h5'

def load_data(data_file_path):
    print("START LOADING H5")
    h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), data_file_path), "r")


    print("len(h5f_data) = ",len(h5f_data))
    np_data = np.zeros((len(h5f_data), 128, 128), dtype=np.uint8)

    for i in tqdm(range(len(h5f_data))):
        dset = h5f_data["data_"+str(i)]
        np_data[i] = np.asarray(dset[:])
        
    return np_data

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------------------------------------
    model_R = Model_R().eval()
    model_S = Model_S().eval()
    
    model_R.load_state_dict(torch.load("checkpoints/1_model.pt"))
    model_S.load_state_dict(torch.load("checkpoints/0_model.pt"))

    model_R = model_R.to(device)
    model_S = model_S.to(device)

#-------------------------------------
    model_D = Model_D()
    model_M = Model_M().eval()

    model_D.load_state_dict(torch.load("checkpoints/discriminator_model.pt"))
    model_M.load_state_dict(torch.load("checkpoints/mapping_model.pt"))
    #print(model_M)
    
    model_D = model_D.to(device)
    model_M = model_M.to(device)
    
    real = load_data(real_path)
    sim = load_data(sim_path)
    
    for i in range(len(real)):
     
        cv2.imshow("origin:",real[i])
        cv2.imshow("gt:",sim[i])

        img = real[i] /255
        img = img.reshape((1,1,128,128))
        img = torch.tensor(img, dtype=torch.float32).to(device)

        mu,_ = model_R.encoder(img)
        mu_sim = model_M(mu)
        recon_imgs = model_S.decoder(mu_sim).cpu().detach().numpy()

        recon_imgs = recon_imgs.reshape((128,128))
        recon_imgs = recon_imgs*255
        
        cv2.imshow("recon:",recon_imgs)
        time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.waitKey(0)



