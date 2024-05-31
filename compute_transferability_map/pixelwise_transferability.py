from logging import raiseExceptions
from random import sample, shuffle
import numpy as np
import ot
import geomloss
import torch
import math
import os 
from optparse import OptionParser
import torch.nn.functional as F

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0,"../compute_model_transferability/")
from OTCE import compute_ce
from utils.LogME import LogME


def compute_coupling(X_src, X_tar):
    
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(X_src,X_tar).numpy()
    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C, numItermax=3000000)
    W = np.sum(P*np.array(C))

    return P,W


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

    parser.add_option('--gpu_id', dest='gpu_id',type='int',
                      default=0, help='gpu id ')

    parser.add_option('--result_file', dest='result_file',type='str',
                      default=None, help='result file')


    parser.add_option('--sampling_strategy', dest='sampling_strategy', type='str',
                      default='false', help='sampling strategy')

    parser.add_option('--edge_type', dest='edge_type', type='str',
                      default=None, help='edge type')

    parser.add_option('--transfer_setting', dest='transfer_setting', type='str',
                      default=None, help='transfer setting')

    parser.add_option('--metric', dest='metric', type='str',
                      default='LEEP', help='transferability metric')
    
    parser.add_option('--stride', dest='stride',type='int',
                      default=4, help='stride of transferability map')

    (options, args) = parser.parse_args()
    return options



def str2bool(s):
    
    if s == 'False' or s == 'false':
        return False
    elif s == 'True' or s == 'true':
        return True
    else:
        raise Exception('s should be string: True, true or false, False')


def save_probability_matrix(P_z, P_yz, P_y_given_z, task_dir, file_name):
    
    dir_list = ['P_z', 'P_yz', 'P_y_given_z']
    prob_list = [P_z, P_yz, P_y_given_z]
    
    for i, dir_ in enumerate(dir_list):
        cur_path = os.path.join(task_dir, dir_)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

        np.save(os.path.join(cur_path,file_name),prob_list[i])
        

def LEEP(X_tar, Y_tar,num_category=19, h_idx=None, w_idx=None):
    
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
        
        # avoid NaN results
        if P_z[i] == 0:
            continue
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

    # save_probability_matrix(P_z, P_yz, P_y_given_z, task_dir="./trf_maps/probability_matrix/segnet_gta5_2021-08-30-to-aachen", file_name = "%s_%s_%.4f.npy"%(h_idx, w_idx, leep_score))

    return leep_score


def load_npy_data(feature_map_dir, label_map_dir, num_sample=None, edge_type=None, img_path_list=None, dataset=None):

    rdm = np.random.RandomState(2021)
    file_list = os.listdir(feature_map_dir)

    if num_sample is None:
        NUM_SAMPLES = len(file_list)
        selected_samples = file_list
    else:
        NUM_SAMPLES = num_sample
        selected_samples = rdm.choice(file_list, NUM_SAMPLES)

    start = time.time()
    x = []
    y = []

    for i,file in enumerate(tqdm(selected_samples)):
        map = np.load(os.path.join(feature_map_dir, file)).transpose(1,2,0)
        H,W,C = map.shape
        label = np.load(os.path.join(label_map_dir, file))
        x.append(map)
        y.append(label)

    x = np.array(x)
    y = np.array(y)
    
    return x,y


def compute_pixel_transferability(args, tar_x, tar_y, src_x=None, src_y=None, is_weight_map=False):

    rdm = np.random.RandomState(2021)

    total_score = 0.0
    start = time.time()

    N,H,W,C = tar_x.shape
    stride = args.stride
    trf_map = np.zeros((int(H / stride), int(W / stride)))
    
    if args.metric == 'LEEP':
        
        tar_x_tensor = torch.from_numpy(tar_x)
        tar_x = F.softmax(tar_x_tensor,dim=3).numpy()

    if args.transfer_setting == 'ADE20K':
        num_category = 19 #150
    else:
        num_category = 19
        
    logme = LogME(regression=False)

    print ("computing pixel-wise (patch-wise) transferability...")
    for i in tqdm(range(int(H / stride))):
        for j in tqdm(range(int(W / stride))):
            
            cur_patch_x = tar_x[:,i*stride:(i+1)*stride,j*stride:(j+1)*stride,:].reshape(-1,C)
            cur_patch_y = tar_y[:,i*stride:(i+1)*stride,j*stride:(j+1)*stride].reshape(-1)
            
            # filter 250
            ignore_idx = np.where(cur_patch_y==250)
            cur_patch_y_filted = np.delete(cur_patch_y, ignore_idx)
            cur_patch_x_filted = np.delete(cur_patch_x,ignore_idx[0], axis=0)
            
            if cur_patch_x_filted.shape[0] < 5:
                continue
            
            if args.metric == 'LEEP':
                score = LEEP(cur_patch_x_filted, cur_patch_y_filted,num_category=num_category, h_idx=i, w_idx=j)
                
                if is_weight_map:
                    score = np.exp(-score)
                else:
                    score = np.exp(score)
                
            elif args.metric == 'LogME':
                score = logme.fit(cur_patch_x_filted, cur_patch_y_filted)

            elif args.metric == 'OTCE':

                _,Hs,Ws,Cs = src_x.shape
                
                if Cs != C:
                    raise Exception("src feature and target feature should have the same dimension")

                H_ratio = Hs / H
                W_ratio = Ws / W
                
                i_s = int(round(i * H_ratio))
                j_s = int(round(j * W_ratio))
                
                src_patch_x = src_x[:,i_s*stride:(i_s+1)*stride,j_s*stride:(j_s+1)*stride,:].reshape(-1,Cs)
                src_patch_y = src_y[:,i_s*stride:(i_s+1)*stride,j_s*stride:(j_s+1)*stride].reshape(-1)
                
                # filter 250
                src_ignore_idx = np.where(src_patch_y==250)
                src_patch_y_filted = np.delete(src_patch_y, src_ignore_idx)
                src_patch_x_filted = np.delete(src_patch_x, src_ignore_idx[0], axis=0)
                
                if src_patch_x_filted.shape[0] < 5:
                    continue
                
                src_patch_x_filted_tensor = torch.from_numpy(src_patch_x_filted)
                cur_patch_x_filted_tensor = torch.from_numpy(cur_patch_x_filted)

                P, W_dis = compute_coupling(src_patch_x_filted_tensor, cur_patch_x_filted_tensor)
                score = - compute_ce(P,src_patch_y_filted, cur_patch_y_filted)
            
            
            trf_map[i,j] = score
                
    end = time.time()
    print ("time: %.2f"%(end-start))
    return trf_map





if __name__ == '__main__':

    args = get_args()
    
    tar_feature_name = 'final'

    if args.transfer_setting == 'BDD100K':

        src_task_list = ["segnet_bdd100k_2021-08-27"]
        tar_task_list = ['aachen']

        for src_task in src_task_list:
            src_x = None
            src_y = None
            if args.metric == 'OTCE':
                print ("loading source features")
                src_feature_map_dir = '../feature/source_feature/%s/final/'%(src_task)
                src_label_map_dir = '../feature/source_feature/%s/label/'%(src_task)
                src_x, src_y = load_npy_data(src_feature_map_dir, src_label_map_dir)

            for tar_task in tar_task_list:

                # "target feature" or "target feature from finetuned model"
                tar_feature_map_dir = '../feature/target_feature/%s/%s/%s/'%(src_task, tar_task, tar_feature_name)
                tar_label_map_dir = '../feature/target_feature/%s/%s/label/'%(src_task, tar_task)

                print ("loading target features")
                tar_x, tar_y = load_npy_data(tar_feature_map_dir, tar_label_map_dir,  edge_type= args.edge_type)

                print ("----------{} to {}----------------".format(src_task, tar_task))
                       
                trf_map = compute_pixel_transferability(args, tar_x, tar_y,src_x, src_y)

                save_dir = "./trf_maps/%s"%(args.metric)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                np.save(os.path.join(save_dir, "{}-to-{}.npy".format(src_task,tar_task)), trf_map)
                
                plt.title("{}-to-{}".format(src_task,tar_task))
                plt.imshow(trf_map)            
                plt.savefig(os.path.join(save_dir,"{}-to-{}.png".format(src_task,tar_task)), dpi=300)
    
    else:
        raise Exception('unconfigured setting')
