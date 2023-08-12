import h5py
import numpy as np
import os
from tqdm import tqdm
import cv2
import time

def load_data(data_file_path,batch_size,img_size):
    print("START LOADING H5")
    h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), data_file_path), "r")

    batch_num = int(len(h5f_data) / batch_size)

    print("len(h5f_data) = ",len(h5f_data))
    np_data = np.zeros((len(h5f_data), img_size, img_size), dtype=np.uint8)

    for i in tqdm(range(len(h5f_data))):
        dset = h5f_data["data_"+str(i)]
        np_data[i] = np.asarray(dset[:])
        
    for i in range(len(np_data)):
        cv2.imshow("data",np_data[i])
        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyWindow("data")

data_file_path = "/home/braynt/Tac/tactile_gym/tactile_gym/vae_trans/data/tactile_pair/tactile1/data_vae_pair.h5"   #/home/braynt/Tac/tactile_gym/tactile_gym/vae_trans/data/tactile/tactile0/data_vae_all.h5"    #/home/braynt/Tac/tactile_gym/tactile_gym/vae_trans/data_collect/sim_unpaired_1.h5
batch_size = 32
img_size = 128
load_data(data_file_path, batch_size, img_size)

