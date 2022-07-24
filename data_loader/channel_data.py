import torch
import numpy as np
from scipy import ndimage
import torchvision
import pandas as pd
import matplotlib.pyplot as plt



class channel_dataset(torch.utils.data.Dataset):
    def __init__(self,mode,seed,snr,size):
        np.random.seed(seed)
        obj = pd.read_pickle(r'data_loader/normalData10_pkl_'+str(size)+'_SNR_'+str(snr))
        ID_samples=np.vstack(obj.ID.values/255.)
        OD_samples = np.vstack(obj.OD.values/255.)
        self.X =  np.reshape(ID_samples,(ID_samples.shape[0], -1))
        if mode=='val':
            self.X = np.reshape(OD_samples, (OD_samples.shape[0], -1))
        elif mode == 'test':
            ID_samples = np.vstack(obj.ID.values / 255.)
            OD_samples = np.vstack(obj.OD.values / 255.)
            '''
            for i in range(0,5):
                fig, axs = plt.subplots(2, 10)
                for j in range(0, 10):
                    axs[0, j].imshow(ID_samples[i*10+j, :, :])
                    axs[1, j].imshow(OD_samples[i*10+j, :, :])
                plt.show()'''
            self.X = np.concatenate((OD_samples,ID_samples),axis=0)
            self.X = np.reshape(self.X, (self.X.shape[0], -1))
            self.OOD = np.ones(self.X.shape[0])
            self.OOD[0:OD_samples.shape[0]] = 0


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index,:]
