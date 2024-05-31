from logging import raiseExceptions
from random import sample, shuffle
import numpy as np
import torch
import os 
from optparse import OptionParser
import torch.nn.functional as F
import time
from tqdm import tqdm


def compute_coupling(X_src, X_tar):
    
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(X_src,X_tar)
    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C, numItermax=3000000)
    W = np.sum(P*np.array(C.numpy()))

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
                      default=1, help='number of repeat times')

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

    (options, args) = parser.parse_args()
    return options



def str2bool(s):
    
    if s == 'False' or s == 'false':
        return False
    elif s == 'True' or s == 'true':
        return True
    else:
        raise Exception('s should be string: True, true or false, False')



def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
    

def getDiffNN(f,Z, rcond=1e-9):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z.reshape((-1,))))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[np.reshape(Z==z, (-1,))], axis=0)
        g[np.reshape(Z==z, (-1,))]=Ef_z
    
    Covg=getCov(g)

    if len(alphabetZ) == 1:
        Covg = np.eye(Covg.shape[0]) / (19 * 19)

    # print ("inverse covf", np.linalg.pinv(Covf,rcond=rcond))
    # print ("covg", Covg)
    dif=np.trace(np.dot(np.linalg.pinv(Covf,rcond=rcond), Covg))
    
    return dif



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
    x = None
    y = None
    for i,file in enumerate(tqdm(selected_samples)):
        map = np.load(os.path.join(feature_map_dir, file))

        C,H,W = map.shape
        f = torch.from_numpy(map)

        f = f.transpose(0, 1).transpose(1, 2).contiguous().view(-1, C).numpy()
        label = np.load(os.path.join(label_map_dir, file)).reshape(-1)
        
        sample_id = file.split('.')[0]

        if img_path_list is not None:
            img_file_path = img_path_list[sample_id]

       
        # filter 250
        ignore_idx = np.where(label==250)
        label_filted = np.delete(label, ignore_idx)
        f_filted = np.delete(f,ignore_idx[0], axis=0)

        if x is None:
            x = f_filted
        else:
            x = np.concatenate((x, f_filted))
        
        if y is None:
            y = label_filted
        else:
            y = np.concatenate((y, label_filted))


    return x,y

def compute_hscore(args, tar_x, tar_y, rcond=1e-9):

    rdm = np.random.RandomState(2021)

    total_score = 0.0
    start = time.time()
    NUM_REPEAT = args.num_repeat
    NUM_PIXEL = args.num_pixel

    tar_x_tensor = torch.from_numpy(tar_x)
    tar_x = F.softmax(tar_x_tensor,dim=1).numpy()

    for i in range(NUM_REPEAT):

        #sampling target data
        
        if NUM_PIXEL is None:
            
            tar_x_sampled = tar_x
            tar_y_sampled = tar_y
            
        else: 
                
            if args.sampling_strategy == 'random':
                print ("random sampling %d"%(NUM_PIXEL))
                tar_pixel_idx_list = np.linspace(0,tar_x.shape[0]-1, tar_x.shape[0]).astype(np.int)
                tar_sampled_pixel_idx = rdm.choice(tar_pixel_idx_list, NUM_PIXEL,replace=False)
                
            else:
                raise Exception("specify sampling strategy: edge, random, random-per-class")


            tar_x_sampled = tar_x[tar_sampled_pixel_idx]
            tar_y_sampled = tar_y[tar_sampled_pixel_idx]        

        score = getDiffNN(tar_x_sampled, tar_y_sampled, rcond=rcond).astype(np.float)

        total_score += score

    end = time.time()
    print ('time: %.4f,  H-score: %.4f'%(end-start,total_score / NUM_REPEAT))
    return total_score / NUM_REPEAT



if __name__ == '__main__':

    args = get_args()
    rcond = 1e-5
    tar_feature_map_dir = args.tar_predicted_map_dir
    tar_label_map_dir = args.tar_gt_map_dir

    print ("loading target features")
    tar_x, tar_y = load_npy_data(tar_feature_map_dir, tar_label_map_dir)   
    
    score = compute_hscore(args, tar_x, tar_y, rcond=rcond)



