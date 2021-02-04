#%%
import os
import csv
import json
import time
import pdb
import ast
import numpy as np
import pandas as pd
from monai.data import partition_dataset_classes,partition_dataset,CacheDataset
from monai.transforms import *
#%%
datapath = "/content/drive/My Drive/Work_from_home/BRATS_segmentation/BRATS2017/Brats17TrainingData/"
details = {"dest":"data_directory"}
#%% Data setup
def setup():
    with open('data_directory/filelist.csv','w', newline='') as file:
        class_names = os.listdir(datapath)
        file_names = []
        label_values = []
        for i in range(len(class_names)):
            data_path = datapath+class_names[i]
            folder_names = os.listdir(data_path)
            for j in range(len(folder_names)):
                file_names.append({'img':data_path+"/"+folder_names[j]+"/"+folder_names[j]+"_t1.nii.gz",'label':i})
                label_values.append(i)
        writer = csv.writer(file)
        for i in range(len(file_names)):
            writer.writerow([file_names[i], label_values[i]])
        dsinfo = {"classes":class_names,"shape":[240,240,155],"count":len(label_values),"prepared":time.asctime()}
        json.dump(dsinfo,open('%s/info.json'%details["dest"],'wt'))
#%%
def DataHelper(cfg = {}):
    # print('in datahelper cfg:',cfg)
    partitionargs = {}
    if 'partition' in cfg:
        partitionargs = cfg['partition']
    idx = partition(partitionargs)
    df = pd.read_csv('%s/filelist.csv'%details['dest'],header = None)
    X,_ = df[0],df[1]
    X = X.values.tolist()
    train_names = [ast.literal_eval(X[i]) for i in idx["train"]]
    valid_names = [ast.literal_eval(X[i]) for i in idx["val"]]
    test_names = [ast.literal_eval(X[i]) for i in idx["test"]]
    transforms = default_transforms()
    print("\nReading training data")
    train_data = CacheDataset(train_names,transform = transforms["train"])
    print("\nReading validation data")
    valid_data = CacheDataset(valid_names,transform = transforms["valid"])
    print("\nReading test data")
    test_data = CacheDataset(test_names,transform = transforms["test"])
    print("\n")
    return {"train":train_data,"valid":valid_data,"test":test_data}
#%%
def _sample_idxlist(idxlist,frac):
    c = np.array(idxlist).copy()
    np.random.shuffle(c)
    return c[:int(len(idxlist)*frac)].tolist()
#%%
def partition(kwargs): 
    # stratified=False, ratios=[0.7,0.2,0.1],shuffle=True, seed=0):
    # updated signature for consistency with monai.data 
    # https://docs.monai.io/en/latest/data.html#partition-dataset
    dest = details['dest']
    info = json.load(open('%s/info.json'%dest))
    cnt = info["count"]
    idx = {'train':[],'val':[],'test':[]}
    if len(kwargs) == 0:
        kwargs['stratified'] = False
        kwargs['ratios'] = [0.7,0.2,0.1]
        kwargs['shuffle'] = True
    frac = 1.
    if 'sampling' in kwargs:
        frac = kwargs['sampling']
        del kwargs['sampling']
    allidx = np.arange(cnt)
    df = pd.read_csv('%s/filelist.csv'%dest,header=None)
    X,y = df[0],df[1]
    if 'stratified' in kwargs:
        del kwargs['stratified']

        idx['train'],idx['val'],idx['test'] = tuple(
            partition_dataset_classes(allidx,
                y.values.tolist(),
                **kwargs))
    #     skf = StratifiedKFold(n_splits=1)
    #     idx['train'],rest=skf.split(X,y)
    #     Xv = X.iloc[rest]
    #     yv = y.iloc[rest]
    #     val2_idx, test2_idx = skf.split(Xv,yv)
    #     idx['val']=rest[val2_idx]
    #     idx['test']=rest[test2_idx]
    else:
        idx['train'],idx['val'],idx['test'] = tuple(
            partition_dataset(allidx,**kwargs))
    #     allidx = np.arange(cnt)
    #     np.random.shuffle(allidx)
    #     ntest = int(test_frac*cnt)
    #     nval = int(val_frac*cnt)
    #     ntrain = cnt - ntest - nval
    #     idx['train']=allidx[:ntrain]
    #     idx['val']=allidx[ntrain:ntrain+nval]
    #     idx['test']=allidx[ntrain+nval:]
    for key in 'train','val','test':
        idx[key] =_sample_idxlist(idx[key],frac)
    return idx
#%%
def default_transforms():
    net_orientation = "RAS"
    vol_resize = [128,128,128]
    train_transform=Compose([
                            LoadNiftiD(keys = "img"),
                            AddChannelD(keys = "img"),
                            # SpacingD(KEYS, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
                            OrientationD(keys = "img", axcodes = net_orientation),
                            ScaleIntensityD(keys = "img"),
                            ResizeD(keys = "img", spatial_size = vol_resize),
                            # RandAffineD(KEYS, spatial_size=(-1, -1, -1),rotate_range=(0, 0, np.pi/2),scale_range=(0.1, 0.1),mode=('bilinear', 'nearest'),prob=1.0),
                            ToTensorD(keys = ("img","label")),
                            ])
    valid_transform = train_transform
    # test_transform = train_transform
    return {"train":train_transform, "valid": valid_transform, "test": valid_transform}