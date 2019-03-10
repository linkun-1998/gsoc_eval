#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import os


# In[2]:


os.listdir()


# In[3]:


filename = '1541962108935000000_167_838.h5'


# In[4]:


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


# In[5]:


with h5py.File(filename, 'r') as f:
    for d_path in return_path(filename):
        print('Path:', d_path)
        print('Shape:', f[d_path].shape)
        print('Data type:', f[d_path].dtype)


# In[ ]:





# In[ ]:




