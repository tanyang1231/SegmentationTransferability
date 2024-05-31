from logging import raiseExceptions
from random import sample, shuffle
import numpy as np

import torch
import os 
from optparse import OptionParser
import time
from utils.LogME import LogME
from tqdm import tqdm



def get_args():

    parser = OptionParser()

    #-----------for empirical setting---------------
    parser.add_option('--tar_predicted_map_dir', dest='tar_predicted_map_dir',type='str',
                      default=None, help='target predicted_map_dir')

    parser.add_option('--tar_gt_map_dir', dest='tar_gt_map_dir',type='str',
                      default=None, help='target gt_map_dir')

    parser.add_option('--num_gt_classes', dest='num_gt_classes',type='int',
                      default=None, help='num_gt_classes')

    parser.add_option('--num_pixel', dest='num_pixel',type='int',
                      default=None, help='number of sampled pixels')

    parser.add_option('--num_repeat', dest='num_repeat',type='int',
                      default=None, help='number of repeat times')

    parser.add_option('--result_file', dest='result_file',type='str',
                      default=None, help='result file')

    (options, args) = parser.parse_args()
    return options



def compute_LogME_randomsample():

    args = get_args()

    tar_predicted_map_dir = args.tar_predicted_map_dir
    tar_gt_map_dir = args.tar_gt_map_dir
    
    rdm = np.random.RandomState(2021)

    logme = LogME(regression=False)


    tar_x = None
    tar_y = None
    NUM_TAR_SAMPLES = len(os.listdir(tar_predicted_map_dir))
    for i in tqdm(range(NUM_TAR_SAMPLES)):

        map = np.load(os.path.join(tar_predicted_map_dir, "%d.npy"%i))
        C,H,W = map.shape
        f = torch.from_numpy(map)
        f = f.transpose(0, 1).transpose(1, 2).contiguous().view(-1, C).numpy()
        label = np.load(os.path.join(tar_gt_map_dir, "%d.npy"%i)).reshape(-1)

        # filter 250
        ignore_idx = np.where(label==250)
        label_filted = np.delete(label, ignore_idx)
        f_filted = np.delete(f,ignore_idx[0], axis=0)

        if tar_x is None:
            tar_x = f_filted
        else:
            tar_x = np.concatenate((tar_x, f_filted))
        
        if tar_y is None:
            tar_y = label_filted
        else:
            tar_y = np.concatenate((tar_y, label_filted))
    
    # print ("target feature loaded")

    total_score = 0.0
    start = time.time()
    NUM_REPEAT = args.num_repeat
    NUM_PIXEL = args.num_pixel

    for i in range(NUM_REPEAT):
        
        #sampling target data
        tar_pixel_idx_list = np.linspace(0,tar_x.shape[0]-1, tar_x.shape[0]).astype(np.int)
        tar_sampled_pixel_idx = rdm.choice(tar_pixel_idx_list, NUM_PIXEL,replace=False)
        tar_x_sampled = tar_x[tar_sampled_pixel_idx]
        tar_y_sampled = tar_y[tar_sampled_pixel_idx]        

        score = logme.fit(tar_x_sampled, tar_y_sampled)

        print (i, score)
        total_score += score

    end = time.time()
    print ('time: %.4f, LogME score: %.4f'%(end-start,total_score / NUM_REPEAT))


def compute_LogME():
    
    args = get_args()

    tar_predicted_map_dir = args.tar_predicted_map_dir
    tar_gt_map_dir = args.tar_gt_map_dir
    
    rdm = np.random.RandomState(2021)

    logme = LogME(regression=False)


    tar_x = None
    tar_y = None
    NUM_TAR_SAMPLES = len(os.listdir(tar_predicted_map_dir))
    print ("loading target features")
    for i in tqdm(range(NUM_TAR_SAMPLES)):

        map = np.load(os.path.join(tar_predicted_map_dir, "%d.npy"%i))
        C,H,W = map.shape
        f = torch.from_numpy(map)
        f = f.transpose(0, 1).transpose(1, 2).contiguous().view(-1, C).numpy()
        label = np.load(os.path.join(tar_gt_map_dir, "%d.npy"%i)).reshape(-1)

        # filter 250
        ignore_idx = np.where(label==250)
        label_filted = np.delete(label, ignore_idx)
        f_filted = np.delete(f,ignore_idx[0], axis=0)

        if tar_x is None:
            tar_x = f_filted
        else:
            tar_x = np.concatenate((tar_x, f_filted))
        
        if tar_y is None:
            tar_y = label_filted
        else:
            tar_y = np.concatenate((tar_y, label_filted))
    
    # print ("target feature loaded")
    start = time.time()
    score = logme.fit(tar_x, tar_y)
    end = time.time()
    print ('time: %.4f, LogME score: %.4f'%(end-start,score))



if __name__ == '__main__':
    # compute_LogME_randomsample()
    compute_LogME()