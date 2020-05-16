## Loading the Labels' files

import pandas as pd
import sys

print(sys.argv)
start_train = int(sys.argv[1])
end_train =  int(sys.argv[2])


dir_train_label='../../_DOCS/ref_train.txt'
train_label = pd.read_csv(dir_train_label, sep=" ", header=None)
train_label.columns = ['edf_Name', 'Start_ts_evnt','End_ts_evnt','Label','Conf']




## Create the list that including all the edf files for training and validation datasets
import numpy as np
train_edf = train_label["edf_Name"]

list_train_edf = []
i=0
while i < len(train_edf):
    list_train_edf.append(train_edf.iloc[i])
    i+=1


tcp_ar = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF']
tcp_le = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE',  'EEG CZ-LE']
tcp_ar_a = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG CZ-REF']


# this function helps to find the file by searching through the given path

import os, fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# create the list of edf files path ( train/valid) by using the find function above
#pathy = " " # insert the path of two datasets ( /train & /edv )
pathy = '../../edf/train/'
edf_train_path = []
i=0
while i < len(list_train_edf[start_train:end_train]):
    edf_train_path.append(find(list_train_edf[i]+'.edf',pathy))
    i+=1

edf_train_path =[i for i in edf_train_path if i!=[]]    



## Load and read the edf file by using mne module 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf

## TODO: Create function for reading the edf file, resampling it to 250Hz,...
## save to dataframe, using train_label & valid_label to extract the event time of each label
## (background, pre-seizure and seizure) and return the dic or dataframe with features and labels

def xyz(path,sFreq=250, low_freq=0.5, hi_freq=40):
   # sFreq: sampling frequency = 250 Hz
   # low_freq, hi_freq for bandpass filter to extract the features frequency-based 
   # path = edf_train_path
    df_b = pd.DataFrame()
    df_s = pd.DataFrame()
    df_pz = pd.DataFrame()
    i = 0
    while i < len(path):
        
        raw = mne.io.read_raw_edf((', '.join(path[i])),preload=True,)
        raw = raw.resample(sFreq, npad='auto')
        raw = raw.filter(low_freq, hi_freq, fir_design='firwin', skip_by_annotation='edge')
        raw = raw.to_data_frame() # save to dataframe
        
        ## TODO using train_label & valid_label to extract the event time of each label
## (background, pre-seizure and seizure) and return the dic or dataframe with features and labels
        
        
        if 'EEG FP1-le' in raw.columns:
            j=0 
            feat_b = []
            feat_s = []
            feat_pz= []
            while j < len(tcp_le):
                feat = raw[tcp_le[j]].tolist()
                if train_label['Label'][i] == 'bckg':
                    feat1 = feat[slice(int(train_label['Start_ts_evnt'][i].item())+10*250 , 15*250)]
                    #print(feat1)
                    for x in feat1:
                        feat_b.append(x)
                        
                elif train_label['Label'][i] == 'seiz':
                    feat2 = feat[slice(int(train_label['Start_ts_evnt'][i].item()), int(train_label['Start_ts_evnt'][i].item())+5*250)]
                    for x in feat2:
                        feat_s.append(x)
                    
                    feat3 = feat[slice(int(train_label['Start_ts_evnt'][i].item())-5*250, -int(train_label['Start_ts_evnt'][i].item()))]
                    for x in feat3:
                        feat_pz.append(x)
                j+=1
                                 
            ft_b = pd.DataFrame([feat_b])
            df_b = df_b.append(ft_b)
            ft_s = pd.DataFrame([feat_s])
            df_s = df_s.append(ft_s)
            ft_pz = pd.DataFrame([feat_pz])
            df_pz = df_pz.append(ft_pz)
            
        else:
            
            j=0 
            feat_b = []
            feat_s = []
            feat_pz= []
            while j < len(tcp_ar):
                feat = raw[tcp_ar[j]].tolist()
                if train_label['Label'][i] == 'bckg':
                    feat1 = feat[slice(int(train_label['Start_ts_evnt'][i].item())+10*250 , 15*250)]
                    #print(feat1)
                    for x in feat1:
                        feat_b.append(x)
                        
                elif train_label['Label'][i] == 'seiz':
                    feat2 = feat[slice(int(train_label['Start_ts_evnt'][i].item()),int(train_label['Start_ts_evnt'][i].item())+6*250)]
                    for x in feat2:
                        feat_s.append(x)
                    
                    feat3 = feat[slice(int(train_label['Start_ts_evnt'][i].item())-5*250,-int(train_label['Start_ts_evnt'][i].item()))]
                    for x in feat3:
                        feat_pz.append(x)
                    
                j+=1
                                 
        ft_b = pd.DataFrame([feat_b])
        df_b = df_b.append(ft_b)
        ft_s = pd.DataFrame([feat_s])
        df_s = df_s.append(ft_s)
        ft_pz = pd.DataFrame([feat_pz])
        df_pz = df_pz.append(ft_pz)
        i+=1
    
        #df = feat_b.append(feat_s)
        #df.append(feat_pz)
    return df_b, df_s, df_pz

df_b,df_s,df_pz= xyz(edf_train_path)


# Background Dataframe (features,target)
df_feat_b_train = df_b.dropna()
df_feat_b_train['Label'] = 'bckg'
df_feat_b_train.to_csv('df_feat_b_train_{}-{}.csv'.format(start_train, end_train))

# Seizure Dataframe (features,target)
df_feat_s_train = df_s.dropna()
df_feat_s_train['Label'] = 'seiz'
df_feat_s_train.to_csv('df_feat_s_train_{}-{}.csv'.format(start_train, end_train))

# pre-Seizure Dataframe (features,target)
df_feat_pz_train = df_pz.dropna()
df_feat_pz_train['Label'] = 'pre-seiz'
df_feat_pz_train.to_csv('df_feat_pz_train_{}-{}.csv'.format(start_train, end_train))
