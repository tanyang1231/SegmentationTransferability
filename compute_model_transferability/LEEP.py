from logging import raiseExceptions
from random import sample, shuffle
import numpy as np
import torch
import math
import os 
from optparse import OptionParser
import torch.nn.functional as F

import time
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


def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    """

    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)

    empirical_prediction = pseudo_source_label @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    return leep_score

def cal_LEEP(X_tar, Y_tar,num_category):
    
    NUM_CATEGORY_Z = num_category

    Y_label_set = set(sorted(list(Y_tar.flatten())))
    #num_category_Y = len(Y_label_set)
    num_category_Y = np.max(Y_tar) + 1
    P_yz = np.zeros((num_category_Y, NUM_CATEGORY_Z))
    P_z = np.zeros((NUM_CATEGORY_Z,1))
    num_samples = X_tar.shape[0]
    for i in range(0,num_samples):
          
        P_z[:,0] += X_tar[i]
        P_yz[Y_tar[i],:] += X_tar[i]

    P_z /= num_samples
    P_yz /= num_samples

    P_y_given_z = np.zeros((num_category_Y,NUM_CATEGORY_Z))

    for i in range(0,NUM_CATEGORY_Z):
        P_y_given_z[:,i] = P_yz[:,i] / P_z[i]

    leep_score = 0.0
    for i in range(0, num_samples):
        yi = Y_tar[i]
        xi = X_tar[i]
        p_sum = 0.0
        for j in range(0,NUM_CATEGORY_Z):
            p_sum += (P_y_given_z[yi,j] * xi[j])

        leep_score += math.log(p_sum)

    leep_score /= num_samples

    return leep_score


def compute_LEEP_randomsample():

    args = get_args()

    tar_predicted_map_dir = args.tar_predicted_map_dir
    tar_gt_map_dir = args.tar_gt_map_dir
    
    rdm = np.random.RandomState(2021)

    tar_x = None
    tar_y = None
    NUM_TAR_SAMPLES = len(os.listdir(tar_predicted_map_dir))
    print ("loading target features")
    for i in range(NUM_TAR_SAMPLES):

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
    tar_x_tensor = torch.from_numpy(tar_x)
    tar_x = F.softmax(tar_x_tensor,dim=1).numpy()
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

        # score = cal_LEEP(tar_x_sampled, tar_y_sampled, num_category=19)
        score = LEEP(tar_x_sampled, tar_y_sampled)

        # print (i, score, score2)
        total_score += score

    end = time.time()
    print ('time: %.4f, LEEP score: %.4f'%(end-start,total_score / NUM_REPEAT))


def compute_LEEP():
    
    args = get_args()

    tar_predicted_map_dir = args.tar_predicted_map_dir
    tar_gt_map_dir = args.tar_gt_map_dir
    
    rdm = np.random.RandomState(2021)

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
    
    tar_x_tensor = torch.from_numpy(tar_x)
    tar_x = F.softmax(tar_x_tensor,dim=1).numpy()
    # print ("target feature loaded")

    start = time.time()
    score = LEEP(tar_x, tar_y)
    end = time.time()
    print ('time: %.4f, LEEP score: %.4f'%(end-start,score))


if __name__ == '__main__':
    # compute_LEEP_randomsample()
    compute_LEEP()