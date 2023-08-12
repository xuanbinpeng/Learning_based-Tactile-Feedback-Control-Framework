import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
import time

data_file_path = "../../../jaka/data_new/merge0/"   
img_size = 128
np_data = np.zeros((11000, img_size, img_size), dtype=np.uint8)
f = h5py.File('./data/tactile/tactile0/data_vae_all_pls.h5', 'w')
j = 0
show = True
for data_file in os.walk(data_file_path): 
  for data in data_file[2] :
    h5f_data = h5py.File(os.path.join(data_file_path, data), "r")
    #np_data = np.zeros((len(h5f_data), img_size, img_size), dtype=np.uint8)
    for i in tqdm(range(len(h5f_data))):
      dset = h5f_data["data_"+str(i)]
      np_data[j+i,:,:] = np.asarray(dset[:])
    j = j+len(h5f_data)
 
  
for i in range(5600):
  f.create_dataset('data_'+str(i), data=np_data[i])
  
print("end")

if show == True:
  for i in range(len(np_data)):
    cv2.imshow("data",np_data[i])
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyWindow("data")
else:
  pass
#f.create_dataset('data_'+str(i), data=env.current_img)

