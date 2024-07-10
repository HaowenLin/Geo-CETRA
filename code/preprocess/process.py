
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import pickle
import random
import copy
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import pyarrow.parquet as pq


def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = cos(lat2) * sin(dlon)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    return bearing


def mean_lat_long(data_set):
    points_lat = np.array([])
    points_long = np.array([])
    for key in data_set:
        traj = data_set[key]
        points_lat  = np.append(points_lat ,traj[:,2])
        points_long  = np.append(points_long ,traj[:,3])

    lat_mean = points_lat.mean()
    long_mean = points_long.mean()
    lat_min, lat_max = points_lat.min() , points_lat.max()
    long_min, long_max = points_long.min() , points_long.max()
    print(f'lat mean {lat_mean} long mean {long_mean}')
    print(f'lat min {lat_min} lat max {lat_max} {long_min} long max {long_max}')
    return lat_mean,long_mean




def get_stats(df):
    df['start_datetime']= pd.to_datetime(df['start_datetime'])
    agent_group_size = df.groupby(['agent_id']).size()
    #print(f'number of max traj length = {max(agent_group_size)}')
    early,late =df['start_datetime'].min(), df['start_datetime'].max()
    print(f'earliest recent date {early}  most recent {late}')
    print(f'number of agents {len(agent_group_size)}')

def get_stats_after_process(df):
    # df_list = [pd.DataFrame(df[key], columns=[key]) for key in df]
    # final_df = pd.concat(df_list, axis=1)
    interval_list = []
    traj_length = []
    
    for f in df:
        traj = df[f]
        consecutive = np.diff(traj[:,1])
        interval_list.append(consecutive)
        traj_length.append(traj.shape[0])
    interval_list = np.concatenate(interval_list)
    traj_length = np.array(traj_length)
    

    max_length = traj_length.max()
    min_interval = interval_list.min()
    max_interval = interval_list.max()
    num_traj = len(df)
    print(f'number of  traj = {num_traj}')
    print(f'number of max traj length = {max([df[key].shape[0] for key in df])}')
    print(f'min time interval = {min_interval} max time interval = {max_interval}')
    print(f'traj length max = {max_length} ')
    print(f'total points = {np.sum(traj_length)} ')


    return num_traj, max_length,min_interval,max_interval


def load_orignal_file(data_file):
    # Load the original data
    
    data_load = np.load(data_file,allow_pickle=True) 
    column_values  = ['agent_id', 'latitude', 'longitude','start_datetime', 'stay_time_minutes']
    data_df = pd.DataFrame(data = data_load,   
                    columns = column_values)
    

    #get_stats(data_df) # number of max traj length = 426

    return data_df


def split_by_day(df,time_interval = 60,days_total = 7, start_time ="2023-05-08T00:00:00Z"):
    ''' 
    in original file the time is one week, we split the data into 7 days
    convert the time to minutes by divide by 60mins

    ''' 
    
    
    df['start_datetime']= pd.to_datetime(df['start_datetime'])
    df = df[df["start_datetime"] >= pd.Timestamp(start_time,tz='UTC')] ### make sure start time is later than the start time

    df = df[df["stay_time_minutes"] >= 10] ### make sure stay duration is longer 
    # basedate = pd.Timestamp((start_time))
    basedate = pd.Timestamp((start_time),tz='UTC')
    df["time"] = df["start_datetime"].apply(lambda x: (x - basedate).total_seconds() / 60./time_interval)
    
    agent_group_size = df.groupby(['agent_id']) 
    sequences = {}
    one_day_min = (24*60/time_interval)
    min_size = math.inf
    max_size = -math.inf
    for name,agent_group in agent_group_size:
        for days in range(days_total):
            
            date = basedate + pd.Timedelta(days=days) 
            seq_name = f"{name}_"+ f"{date.day:02d}"
            #print(seq_name)
            start = (date - basedate).days * one_day_min
            end = (date + pd.Timedelta(days=1) - basedate).days *one_day_min
            df_range = agent_group[agent_group["time"] >= start]
            df_range = df_range[df_range["time"] < end]
            df_range["time"] = df_range["time"] - start
            df_range["time"] = df_range["time"].astype(np.float64)
            df_range["stay_time_minutes"] = (df_range["stay_time_minutes"] / time_interval).astype(np.float64)
            seq = df_range[['agent_id','time', 'latitude', 'longitude', 'stay_time_minutes']].to_numpy().astype(np.float64)

            length = seq.shape[0]
            # min_size = min(min_size,length)
            # max_size = max(max_size,length)
            if length >=5 and length <100:

                sequences[seq_name] = seq
        
        #break
    return sequences
            

def remove_zero_start(dataset):
    '''
    remove the traj that start from 0
    agent_id, time, latitude, longitude, stay_time_minutes
    '''

    
    
    filter_list = {}

    for f in dataset.keys():
        traj = dataset[f]    
        #print(traj.shape)
        if traj[0][1] == 0: continue
        filter_list[f] = traj
    return filter_list

def get_polar(dataset):
    
    


    lat_mean,long_mean = mean_lat_long(dataset)

    filter_list = {}
    for f in dataset:
        traj = dataset[f]    
    

        distances = np.zeros(traj.shape[0])
        angles = np.zeros(traj.shape[0])
        points = traj[:,2:4]

        points = np.concatenate((np.array([lat_mean,long_mean]).reshape(1,-2),points),axis = 0)
        for j, (p1,p2) in enumerate(zip(points[:-1,:],points[1:,:])):
            distance = haversine(p1[0],p1[1],p2[0],p2[1])
            angle = calculate_bearing(p1[0],p1[1],p2[0],p2[1])
            distances[j] = distance
            angles[j] = angle
            
        
        cumsum = np.sum(distances[1:]) 
        if cumsum < 1e-2: continue #### filter out trajectories without moving
    
        traj = np.insert(traj,4,distances,axis = 1)
        traj = np.insert(traj,5,angles,axis = 1)
        # traj[:,2] = distances
        # traj[:,3] = angles

        ##### caucious filter out stop points here 
        save_traj = traj[traj[:,2]!= 0]
        if save_traj.shape[0] < 5: continue #### filter out trajectories are too short 

        filter_list[f] =save_traj
        
    
    return filter_list


def random_generate_constraint(dataset,min_val = 0.002,constant_large = 24.0):


    
    filter_list = {} 
    traj_num  = len(dataset)



    constraint_sample_save = np.zeros((traj_num,3))
    for traj_id,f in enumerate(dataset):
        traj = dataset[f] 
        traj_length = traj.shape[0]
        constrain_index_range = range(1,traj_length)  
        constraint_index = random.sample(constrain_index_range,1)[0] #sample index
        #print(constraint_index)
        cur_lower_time = random.uniform(traj[constraint_index - 1][1],traj[constraint_index][1])
        #print(traj[constraint_index - 1][1],traj[constraint_index ][1],traj[constraint_index + 1][1])
        if constraint_index == traj_length - 1: # last index 

            cur_upper_time = random.uniform(traj[constraint_index][1],24)
        else:
            cur_upper_time = random.uniform(traj[constraint_index][1],traj[constraint_index+1][1])
        #print(traj[:,1])
        #print(constraint_index,cur_lower_time,cur_upper_time)
        prev_upper_time = 0.0
        #print(traj)

        time_bound = np.zeros((traj.shape[0],2))
        time_bound[:,0] = min_val
        time_bound[:,1] = constant_large
        

        ### below save three readable constraint
        constraint_sample_save[traj_id,0] = constraint_index
        constraint_sample_save[traj_id,1] =cur_lower_time
        constraint_sample_save[traj_id,2] =cur_upper_time

        for i in range(0,constraint_index):
            #print(i,traj[i][-2])
            time_bound[i,-2] = min_val
            time_bound[i,-1] = cur_lower_time - prev_upper_time
            prev_upper_time = traj[i,1]
        time_bound[constraint_index,-2] = cur_lower_time-prev_upper_time ### exact at constraint loc
        time_bound[constraint_index,-1] = cur_upper_time-prev_upper_time
        if constraint_index != traj_length-1:
            time_bound[constraint_index+1,-2] = cur_upper_time - traj[constraint_index,1]

        traj= np.concatenate((traj,time_bound),axis = 1)

        for row in time_bound:
            assert row[-1] > row[-2], print(f"traj {traj_id} Row {traj} does not satisfy the condition that the 7th column is greater than the 6th column")
        filter_list[f] =traj

    
    
    
    print('end process')
    return filter_list,constraint_sample_save

def get_statis_random_for_generation(traj_dataset,readable_constraint,max_length):
    
    '''
    remove the traj that start from 0
    agent_id, time, latitude, longitude, distance,angle, stay_time_minutes, lower_time, upper_time
    '''
 
    
    dataset = readable_constraint
    #print(readable_constraint)
    
    all_trajs = [traj_dataset[f] for f in traj_dataset]
    #save_statics = np.zeros((dataset.shape[0],max_length))
    assert dataset.shape[0] == len(all_trajs), print(f"traj num does not match ")
    count_list = np.zeros((dataset.shape[0],max_length))
    for i in range(dataset.shape[0]):   
        cur_lower_time, cur_upper_time = dataset[i,1],dataset[i,2]
        for id,traj in enumerate(all_trajs):
            for point_index in range(1,traj.shape[0]):
                time = traj[point_index,1]
                if cur_lower_time < time and time < cur_upper_time:
                    count_list[i,point_index] +=1 
    
    
    return count_list





def houston_process():
    dataset = load_houston() ## load the data
    dataset = remove_zero_start(dataset) 

    
    dataset = get_polar(dataset)
    num_traj, max_length,min_interval,max_interval = get_stats_after_process(dataset)

    dataset,constraint_sample_save = random_generate_constraint(dataset,min_val = 0.002,constant_large = 24.0) ### generate realistic contriaints 
    count_list = get_statis_random_for_generation(traj_dataset= dataset,readable_constraint= constraint_sample_save,max_length=max_length)
    dir_name =  '/home/users/constraint_gen/dataset/veraset/houston/my_process/'
    traj_dataset_loc = dir_name + 'houston_60mins_random_constraint_traj'
    readable_constraint_loc = dir_name + 'houston_60mins_readable_constraint'
    count_list_loc = dir_name + 'houston_60mins_statistics_constraint'
    
    np.savez(traj_dataset_loc, **dataset)
    np.save(readable_constraint_loc,constraint_sample_save)
    np.save(count_list_loc,count_list)

def load_houston(time_interval = 60):

    #agent_id, time, latitude, longitude, stay_time_minutes

    input_file = '/storage/dataset/houston/raw/veraset_data_march_05_06_07_08_09_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_Houston_F50_spd_500_0.1.csv'
    
    
    
    
    #start_time_ts = 1584255600 # this is 2020/03/15 00:00:00

    start_time_ts_list = [1584230401,1584342000,1584428400,1584514800,1584601200,1584687600,1584774000]
    time_str = ['0315','0316','0317','0318','0319','0320','0321']
    sequences = {}
    for i in range(len(start_time_ts_list)):
        df = pd.read_csv(input_file, sep=',',names=['user_id', 'id','lat', 'long', 'start_ts','end_ts'])
        start_time_ts = start_time_ts_list[i]
        cur_day = time_str
    
        df = df[df['start_ts'] > start_time_ts]
        max_day  = 1 * 24 
        df['time'] = (df['start_ts'] - start_time_ts) / (60.0 * time_interval) # convert to hours
        df['stay_time_minutes'] = (df['end_ts'] - df['start_ts']) / (60.0 * time_interval) # convert to hours
        
        min_range = 10.0/ 60.0
        df = df[df["stay_time_minutes"] >= min_range]
        
        df = df.drop(columns=['id','start_ts','end_ts'])
        agent_group_size = df.groupby(['user_id']) 
    
        for name,agent_group in agent_group_size:
            agent_group_size = df.groupby(['user_id']) 
        
        
        for name,agent_group in agent_group_size:
            
            seq = agent_group[['user_id','time', 'lat', 'long', 'stay_time_minutes']].to_numpy().astype(np.float64)
            condition = seq[:, 1] < max_day

            seq = seq[condition]
            length = seq.shape[0]
    
            if length >=5 and length <100:

                sequences[f'{cur_day}_{name}'] = seq
            
    return sequences




    

    
def main():
    print('start process')
    houston_process()






if __name__==   '__main__':
    main()