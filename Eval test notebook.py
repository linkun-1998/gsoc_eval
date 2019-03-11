#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pytz
import h5py
from scipy.signal import medfilt


# In[2]:


os.listdir()


# In[3]:


f_name = '1541962108935000000_167_838.h5' #file name in string f_name
ns = int(f_name[:18]) #first 18 digit of the file


# In[4]:


#convert name of the file to given timezones
def convert_nanotime(ns):
    naive_datetime = datetime.fromtimestamp(ns/1e9)
    local_timezone_name = 'Europe/Zurich' #time zone name
    local_timezone = pytz.timezone(local_timezone_name) 
    local_datetime = local_timezone.localize(naive_datetime, is_dst=None) #local datetime
    utc_datetime = local_datetime.astimezone(pytz.utc) #utc datetime
    #for nanosecond precision
    #naive_ns = '{}{:03.0f}'.format(naive_datetime.strftime('%Y-%m-%d %H:%M:%S.%f'), ns%1e3)
    #local_ns = '{}{:03.0f}'.format(local_datetime.strftime('%Y-%m-%d %H:%M:%S.%f'), ns%1e3)
    #utc_ns = '{}{:03.0f}'.format(utc_datetime.strftime('%Y-%m-%d %H:%M:%S.%f'), ns%1e3)
    return local_datetime, utc_datetime #,naive_ns, local_ns, utc_ns

local, utc = convert_nanotime(ns)
print('CERN local datetime (Europe/Zurich timezone): {} '.format(local))
print('UTC datetime: {}'.format(utc))


# In[5]:


#traversing through the datasets
#yield a path until the path dosen't correspond to a dataset
def return_path(filename):    
    def traverse(group, prefix=''):
        for key in group.keys():
            item = group[key]
            path = '{0}/{1}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # If the item is a dataset
                yield(path, item)
            elif isinstance(item, h5py.Group): # if the item is a group
                yield from traverse(item, path)     
    with h5py.File(filename, 'r') as f:
        for path, _ in traverse(f):
            yield path

#Appending the path, shape, size, dtype to list
dataset_path = []
dataset_shape = []
dataset_size = []
#dataset_dtype = []
'''The dtype here is not considered due to the following error.
(No NumPy equivalent for TypeBitfieldID exists)
Please look the other ERROR.ipynb python notebook for more details'''
with h5py.File(f_name, 'r') as f:
    for d_path in return_path(f_name):
        dataset_path.append(d_path)
        dataset_shape.append(f[d_path].shape)
        dataset_size.append(f[d_path].size)
        #dtype.append(f[d_path].dtype)


# In[6]:


#1st approachg for represnting the hdf in csv file
#Converting the list to a CSV file using pandas
def csv_file(pathlist, shapelist, sizelist): #dtypelist)
    df_dict = {}
    grouplist = []
    datasetlist = []
    for path in pathlist:
        temp = list(path.split('/'))
        grouplist.append('/'.join(temp[:-1])) # Separating the group from the path
        datasetlist.append(temp[-1]) #Separating the dataset from the path
    #appending the list to dictionary
    df_dict['group'] = grouplist
    df_dict['dataset'] = datasetlist
    df_dict['shape'] = shapelist    
    df_dict['size'] = sizelist    
    #df_dict['dtype'] = dtypelist
    return pd.DataFrame.from_dict(df_dict)

df = csv_file(dataset_path, dataset_shape, dataset_size )#,dtype)
df.to_csv('csv_file.csv') 
df.head(10)


# In[7]:


'''The csv_file_2 splits the whole path of the dataset individually to groups
sub_groups, and later sub_sub_groups. This will give a csv file more descriptive 
leading to easy query '''
def csv_file_2(pathlist, shapelist, sizelist): #dtypelist)
    dflist = []
    for path, shape, size in zip(dataset_path, dataset_shape, dataset_size):
        lis = list(path.split('/'))[1:]
        temp_dict = {}
        for i, val in enumerate(lis):
            if i!=len(lis)-1:
                temp_dict['sub_'*i+'group'] = val
            else:
                temp_dict['dataset'] = val
        temp_dict['shape'] = shape
        temp_dict['size'] = size
        dflist.append(temp_dict)
    df = pd.DataFrame(dflist)
    new_order=[1,4,5,0,2,3] #ordering the csv according to the requirement
    df = df[df.columns[new_order]]   
    return df

df2 = csv_file_2(dataset_path, dataset_shape, dataset_size )#,dtype)
df2.to_csv('split_csv_file.csv') #save to filename 'split_csv_file.csv'
df2.head(10)


# In[8]:


# taking the image data from the group /AwakeEventData/XMPP-STREAK/StreakImage/
with h5py.File(f_name, 'r') as f:
    img_data = np.array(f["/AwakeEventData/XMPP-STREAK/StreakImage/streakImageData"])
    img_height = np.array(f["/AwakeEventData/XMPP-STREAK/StreakImage/streakImageHeight"])
    img_width = np.array(f["/AwakeEventData/XMPP-STREAK/StreakImage/streakImageWidth"])
# reshaping the 1-d array to 2-d
img = img_data.reshape(img_height[0], img_width[0])

#filtering the image with median filter with default kernel size i.e 3
filter_img = medfilt(img, kernel_size=None)
#display image
plt.figure(figsize=(10,10))
plt.imshow(filter_img)
#saving image in png format
plt.savefig('image.png')


# In[ ]:




