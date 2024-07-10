import numpy as np
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.stats

def load_data_dic(dataset):
    # agent_id, time, latitude, longitude, distance,angle, stay_time_minutes, lower_time, upper_time
    if dataset == 'houston_discrete_600' or dataset == 'houston_discrete':
        data_file = '/home/users/constraint_gen/dataset/veraset/houston/my_process/houston_60mins_random_constraint_traj.npz'
        dataset = np.load(data_file)
    else:
        raise ValueError('dataset not found')
    lengths = []
    intervals = []
    all_distances = []
    internal_distance  = []
    gradius = []
    count = 0
    for f in dataset.files[:]:
        traj = dataset[f]
        #print(traj[0])
        count += traj.shape[0]
    
        lengths.append(traj.shape[0])
        diffs = np.diff(traj[:,1])
        intervals = np.concatenate((intervals,diffs))

        distance = 0.0
    
        for i in range(1,traj.shape[0]):
            #print(traj[0])
            cur_move = traj[i,4]
            
            internal_distance.append(cur_move)
            distance += cur_move
        all_distances.append(distance)
        xs = traj[:,2]
        ys = traj[:,3]
        xcenter, ycenter = np.mean(xs), np.mean(ys)
        dxs = xs - xcenter
        dys = ys - ycenter
        rad = [dxs[i]**2 + dys[i]**2 for i in range(traj.shape[0])]
        rad = np.mean(np.array(rad, dtype=float))
        gradius.append(rad)
    return lengths,intervals,all_distances,internal_distance,gradius 
    


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2]) # convert decimal degrees to radians
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.power(np.sin(dlat/2), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon/2), 2)
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def construct_prob_transition_matrix(trajectories, dataset):
    """

    Args:
        trajectories: a dataframe where each row is a sequence of location in grid id

    Returns:
        a transition matrix where each element is the prob of transition from origin to destination

    """
    # change all the elements in the dataframe to int
    #data = trajectories.astype(int)
    data = trajectories
    max_grid_id = 0
    #print(dataset)
    if dataset == 'houston_discrete':
        #max_grid_id = 61887
        max_grid_id = 66757
    else:
        raise ValueError(f'dataset not found {dataset}')
    # initialize a transition matrix
    transition_matrix = np.zeros((max_grid_id + 1, max_grid_id + 1))
    total_count = 0
    # iterate through each row
    for i in range(len(data)):
        # iterate through all the elements in a row
        traj = data[i]
        #print(f'traj shape {traj.shape}')
        for j in range(traj.shape[0] - 1):
            current_location = traj[j,-1]
            next_location = traj[j + 1,-1]
            # since both of them start from 1, so we need to minus 1
            transition_matrix[int(current_location)][int(next_location)] += 1
            total_count += 1
        #break
    # calculate the probability of transition
    transition_matrix = transition_matrix / total_count
    # return the transition matrix
    return transition_matrix





def metric_radius(sequences,lengths):
    all_radii = []
    for seq,length in zip(sequences,lengths):
        points = seq[:length,1:]
        
        #points = np.array([grid_to_latlong[to_2D(i)] for i in seq])
        com = np.mean(points, 0)
        all_radii.append(np.sqrt(np.mean([haversine(i[1], i[0], com[1], com[0])**2 for i in points])))
    return np.array(all_radii)

def metric_distance(sequences,lengths):

    all_distances = []
    index_test = []
    internal_distance  = []
    for traj_index, (seq,length) in enumerate(zip(sequences,lengths)):
        distance = 0.0
        prev_point = seq[0,1:]
        for i in range(1,length):
            cur_point = seq[i,1:]
            cur_move =  haversine(prev_point[1], prev_point[0], cur_point[1], cur_point[0])
            internal_distance.append(cur_move)
            distance += cur_move
            prev_point = cur_point
        if distance ==0.0:
            index_test.append(traj_index)
        all_distances.append(distance)
    return np.array(internal_distance), np.array(all_distances)


def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js
    

def dis_JD_calculated(real_seq, gen_seq,max_val = 1000,max_bin_num =10000  ):
    P = real_seq
    Q = gen_seq

    d1_dist, _ = arr_to_distribution(
        P, 0,max_val, max_bin_num) #
    d2_dist, _ = arr_to_distribution(
        Q, 0, max_val, max_bin_num)
    return  get_js_divergence(d1_dist,d2_dist) #movesim version 
    #return np.round(distance.jensenshannon(d1_dist,d2_dist), 5)


def dis_JD(real_seq, gen_seq, metric_fn):
    P = metric_fn(real_seq)
    Q = metric_fn(gen_seq)
    return np.round(distance.jensenshannon(P,Q), 5)



def load_test_data(time,epoch_num,interval_option=False,dataset='baseline_60mins_remove_zero',time = '08/11/2024'):
    
    
    data_file = f'/home/users/constraint_gen/exps/thp/{dataset}/{time}/generated_samples_epoch_{epoch_num}.npz'
    data = np.load(data_file,allow_pickle=True) 
    x_denormalized = data['gen_samples_denormalized']
    x_normalized = data['gen_samples_normalized']
    length = []
    all_interval = np.array([])
    process_x = []
    for i in range(x_denormalized.shape[0]):
        seq = x_denormalized[i]
        un_denormalized = x_normalized[i]
        index = np.where(seq[:,0]>24)[0]
        
   
        if len(index) != 0:
            index = index[0]
            
            seq[index:,:] =0
            length.append(index)
        else:
            index = seq.shape[0]
            length.append(index)
        
        if interval_option:
            interval = np.array([j-i for i, j in zip(seq[:index-1,0], seq[1:index,0])])
            all_interval = np.concatenate((all_interval,interval))
        process_x.append(seq)

        

    process_x = np.array(process_x)
    return process_x,length,all_interval
    
    #return data

def load_gt(time,epoch_num,test=False,interval_option=False,dataset='baseline_60mins_remove_zero',time = '08/11/2024'):
    if test:
        test_add = 'test/'
    else:
        test_add = ''

    data_file = f'/home/users/constraint_gen/exps/{test_add}thp/{dataset}/{time}/generated_samples_epoch_{epoch_num}_groundtruth.pkl'
    file = open(data_file, 'rb')
    data = pickle.load(file)
    data_ret = []
    file.close()
    length = []
    all_interval = np.array([])
    for batch in data:
        for seq in batch:
            data_ret.append(seq)
            index = np.where(seq[:,1]==0)[0]
            if len(index)==0:
                index = seq.shape[0]
                
                length.append(index)
            else:
                # if index[0]==0:
                #     a=1
                index = index[0]
                length.append(index)
            
            
            if interval_option:
                interval = np.array([j-i for i, j in zip(seq[:index-1,0], seq[1:index,0])])
                all_interval = np.concatenate((all_interval,interval))
    return data_ret,length, all_interval




def evaluate():

    

    
    test_option = False
    interval_option = False
    epoch_num = 249

    gt_data,gt_length, gt_interval  = load_gt(epoch_num,test=test_option,interval_option=interval_option,dataset=dataset)
    generated_data,gen_length,gen_interval = load_test_data(epoch_num,test=test_option,interval_option=interval_option,dataset=dataset) 

   

    interval_option = True


    gt_data,gt_length, gt_interval  = load_gt(epoch_num,test=test_option,interval_option=interval_option,dataset=dataset)


    generated_data,gen_length,gen_interval = load_test_data(epoch_num,test=test_option,interval_option=interval_option,dataset=dataset)

    

    print(f'gt length: {gt_length[:3]}')
    print(f'gen length: {gen_length[:3]}')
    length_jd = dis_JD_calculated(gt_length,gen_length,max_val=21,max_bin_num =21)
    print(f'length: {length_jd}')

    distance_gen_short,distance_gen = metric_distance(generated_data,gen_length)
    distance_real_short, distance_real = metric_distance(gt_data,gt_length)
    distance_jd = dis_JD_calculated(distance_gen,distance_real,max_val=1000)
    rel_distance_jd = dis_JD_calculated(distance_gen_short,distance_real_short,max_val=1000)

    print(f'max distance: {max(distance_real) } ')
    print(f'distance: {distance_jd} rel distance {rel_distance_jd}')
    
    
    r_gen = metric_radius(generated_data,gen_length)
    r_real = metric_radius(gt_data,gt_length)
    radius_jd = dis_JD_calculated(r_real,r_gen,max_val=100)
    print(f'radius: {radius_jd} ')
    duration_jd = dis_JD_calculated(gt_interval,gen_interval,max_val=20,max_bin_num=100)
    print(f'duration: {duration_jd}')




if __name__ == "__main__":


    evaluate()


