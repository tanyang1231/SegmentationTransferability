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

from utils.SinkhornLoss import SinkhornDistance
import time
from tqdm import tqdm

def compute_coupling(X_src, X_tar):
    
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(X_src,X_tar).numpy()
    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C, numItermax=2000000)
    W = np.sum(P*np.array(C))

    return P,W


def get_args():

    parser = OptionParser()

    #-----------for empirical setting---------------
    parser.add_option('--tar_predicted_map_dir', dest='tar_predicted_map_dir',type='str',
                      default=None, help='target predicted_map_dir')

    parser.add_option('--tar_gt_map_dir', dest='tar_gt_map_dir',type='str',
                      default=None, help='target gt_map_dir')

    parser.add_option('--src_predicted_map_dir', dest='src_predicted_map_dir',type='str',
                      default=None, help='source predicted_map_dir')

    parser.add_option('--src_gt_map_dir', dest='src_gt_map_dir',type='str',
                      default=None, help='source gt_map_dir')

    parser.add_option('--num_gt_classes', dest='num_gt_classes',type='int',
                      default=None, help='num_gt_classes')

    parser.add_option('--num_src_samples', dest='num_src_samples',type='int',
                      default=None, help='number of source samples used for computing')

    parser.add_option('--num_pixel', dest='num_pixel',type='int',
                      default=None, help='number of sampled pixels')

    parser.add_option('--num_repeat', dest='num_repeat',type='int',
                      default=None, help='number of repeat times')

    parser.add_option('--gpu_id', dest='gpu_id',type='int',
                      default=0, help='gpu id ')

    parser.add_option('--OT_solver', dest='OT_solver',type='str',
                      default=None, help='OT solver')

    parser.add_option('--result_file', dest='result_file',type='str',
                      default=None, help='result file')

    parser.add_option('--use_softmax', dest='use_softmax', type='str',
                      default='false', help='use_softmax')

    parser.add_option('--use_edge_pixel', dest='use_edge_pixel', type='str',
                      default='false', help='use edge pixel')

    (options, args) = parser.parse_args()
    return options



def str2bool(s):
    
    if s == 'False' or s == 'false':
        return False
    elif s == 'True' or s == 'true':
        return True
    else:
        raise Exception('s should be string: True, true or false, False')

def compute_ce(P, Y_src, Y_tar):

    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    P_src_tar = np.zeros((np.max(Y_src)+1,np.max(Y_tar)+1))

    for y1 in src_label_set:
        y1_idx = np.where(Y_src==y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar==y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

            P_src_tar[y1,y2] = np.sum(P[RR,CC])

    P_src = np.sum(P_src_tar,axis=1)

    entropy = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            
            if P_src_tar[y1,y2] != 0:
                entropy += -(P_src_tar[y1,y2] * math.log(P_src_tar[y1,y2] / P_y1))

    return entropy 


def calSoftConditionalEntropy(P, Y_src, Y_tar):

    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    P_empirical = P

    P_src_tar = np.zeros((np.max(Y_src)+1,np.max(Y_tar)+1))

    print ('length of label set: %d, max label id: %d'%(len(tar_label_set),np.max(Y_tar)+1))
    r,c = P_empirical.shape

    for i in range(0,r):
        for j in range(0,c):
            P_src_tar[Y_src[i], Y_tar[j]] += P_empirical[i,j]

    #P_src_tar /= len(Y_src)
    P_src = np.sum(P_src_tar,axis=1)

    entropy = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            
            if P_src_tar[y1,y2] != 0:
                entropy += -(P_src_tar[y1,y2] * math.log(P_src_tar[y1,y2] / P_y1))

    return entropy 




# Treat all pixels as the instances of a dataset, and then randomly sample pixels from dataset for computing OTCE score.
# Repeat the process above T times to obtain the stabilized OTCE score.
def compute_otce():

    args = get_args()

    # device = torch.device("cuda:%d"%(args.gpu_id) if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    src_predicted_map_dir = args.src_predicted_map_dir
    src_gt_map_dir = args.src_gt_map_dir

    tar_predicted_map_dir = args.tar_predicted_map_dir
    tar_gt_map_dir = args.tar_gt_map_dir

    src_file_list = os.listdir(src_predicted_map_dir)
    
    rdm = np.random.RandomState(2021)

    src_x = None
    src_y = None
    src_y_edge = None
    if args.num_src_samples is None:
        NUM_SRC_SAMPLES = len(src_file_list)
        selected_src_samples = src_file_list
    else:
        NUM_SRC_SAMPLES = args.num_src_samples
        selected_src_samples = rdm.choice(src_file_list, NUM_SRC_SAMPLES)

    print ("loading source features")
    for i,src_file in enumerate(tqdm(selected_src_samples)):
        map = np.load(os.path.join(src_predicted_map_dir, src_file))
        C,H,W = map.shape
        f = torch.from_numpy(map)
        f = f.transpose(0, 1).transpose(1, 2).contiguous().view(-1, C).numpy()
        label = np.load(os.path.join(src_gt_map_dir, src_file))

        label = label.reshape(-1)   

        # filter 250
        ignore_idx = np.where(label==250)
        label_filted = np.delete(label, ignore_idx)
        f_filted = np.delete(f,ignore_idx[0], axis=0)

        if src_x is None:
            src_x = f_filted
        else:
            src_x = np.concatenate((src_x, f_filted))
        
        if src_y is None:
            src_y = label_filted
        else:
            src_y = np.concatenate((src_y, label_filted))
 
    tar_x = None
    tar_y = None
    tar_y_edge = None
    NUM_TAR_SAMPLES = len(os.listdir(tar_predicted_map_dir))

    print ("loading target features")
    for i in tqdm(range(NUM_TAR_SAMPLES)):

        map = np.load(os.path.join(tar_predicted_map_dir, "%d.npy"%i))
        C,H,W = map.shape
        f = torch.from_numpy(map)
        f = f.transpose(0, 1).transpose(1, 2).contiguous().view(-1, C).numpy()
        label = np.load(os.path.join(tar_gt_map_dir, "%d.npy"%i))

        label = label.reshape(-1)

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


    total_ce = 0.0
    start = time.time()
    NUM_REPEAT = args.num_repeat
    NUM_PIXEL = args.num_pixel

    for i in range(NUM_REPEAT):
        
        #sampling source data

        src_pixel_idx_list = np.linspace(0,src_x.shape[0]-1, src_x.shape[0]).astype(np.int32)
        src_sampled_pixel_idx = rdm.choice(src_pixel_idx_list, NUM_PIXEL,replace=False)

        src_x_sampled = src_x[src_sampled_pixel_idx]
        src_y_sampled = src_y[src_sampled_pixel_idx]        

        src_x_sampled_tensor = torch.from_numpy(src_x_sampled)

        #sampling target data
        tar_pixel_idx_list = np.linspace(0,tar_x.shape[0]-1, tar_x.shape[0]).astype(np.int32)
        tar_sampled_pixel_idx = rdm.choice(tar_pixel_idx_list, NUM_PIXEL,replace=False)

        tar_x_sampled = tar_x[tar_sampled_pixel_idx]
        tar_y_sampled = tar_y[tar_sampled_pixel_idx]        

        tar_x_sampled_tensor = torch.from_numpy(tar_x_sampled) 

        if str2bool(args.use_softmax):
            print ("use softmax for normalize")
            src_x_sampled_tensor = F.softmax(src_x_sampled_tensor,dim=1)
            tar_x_sampled_tensor = F.softmax(tar_x_sampled_tensor,dim=1)

        if args.OT_solver == 'emd':
            P, W_dis = compute_coupling(src_x_sampled_tensor, tar_x_sampled_tensor)
        elif args.OT_solver == 'sinkhorn':
            # print (device)
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=10000, reduction=None, device=device)
            with torch.no_grad():
                src_x_sampled_tensor = src_x_sampled_tensor.to(device)
                tar_x_sampled_tensor = tar_x_sampled_tensor.to(device)
                W_dis, P, C = sinkhorn(src_x_sampled_tensor, tar_x_sampled_tensor)
                P = P.cpu().numpy()

        cur_ce = - compute_ce(P,src_y_sampled,tar_y_sampled)
        total_ce += cur_ce

    end = time.time()
    print ('time: %.4f, OTCE score: %.4f'%(end-start,total_ce / NUM_REPEAT))



if __name__ == '__main__':
    compute_otce()