import numpy as np
import nibabel as nib
import math

class MRIGenerator():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=32, shuffle=True, translation=False):
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.cmpt = 0
        self.shuffle = shuffle
        self.translation = translation
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (idx + 1) * self.batch_size 
        #batch_x = self.list_path.iloc[idx * self.batch_size:idx_end, 1]
        #batch_y = self.list_path.iloc[idx * self.batch_size:idx_end, 4]
        
        batch_x = self.list_path.iloc[idx * self.batch_size:idx_end, 0:4]
        batch_y = self.list_path.iloc[idx * self.batch_size:idx_end, 4]
        if (self.translation):
            return [np.stack(matrice_translation(path[1], pixel_max=10, trans_rate=0.5), axis=3) for path in batch_x.iterrows()], batch_y
        else :
            return [np.stack((nib.load(path[1][0]).get_data(), nib.load(path[1][1]).get_data(), nib.load(path[1][2]).get_data(), nib.load(path[1][3]).get_data()), axis=3) for path in batch_x.iterrows()], batch_y
        #return np.expand_dims([nib.load(path).get_data() for path in batch_x], axis=5), batch_y
    
    def __iter__(self):
        if self.cmpt + 2 > self.len :
            self.cmpt = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.cmpt += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getitem__(self.cmpt)
            self.__iter__()
            yield (np.array(batch_input), batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class MRIGenerator_matrix():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=32, shuffle=True, translation=False):
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.cmpt = 0
        self.shuffle = shuffle
        self.translation = translation
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (idx + 1) * self.batch_size 
        batch_x = self.list_path.iloc[idx * self.batch_size:idx_end, 0]
        batch_y = self.list_path.iloc[idx * self.batch_size:idx_end, 4]
        #batch_x = self.list_path.iloc[idx * self.batch_size:idx_end, 0:4]
        #batch_y = self.list_path.iloc[idx * self.batch_size:idx_end, 4]
        
        #if (self.translation):
        #    return [np.stack(matrice_translation(path[1], pixel_max=10, trans_rate=0.5), axis=3) for path in batch_x.iterrows()], batch_y
        #else :
        #    return [np.stack((np.load(path[1][0]+".npy", allow_pickle=True), np.load(path[1][1]+".npy", allow_pickle=True), np.load(path[1][2]+".npy", allow_pickle=True), np.load(path[1][3]+".npy", allow_pickle=True)), axis=3) for path in batch_x.iterrows()], batch_y
        return np.expand_dims([np.load(path+".npy", allow_pickle=True) for path in batch_x], axis=5), batch_y
    
    def __iter__(self):
        if self.cmpt + 2 > self.len :
            self.cmpt = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.cmpt += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getitem__(self.cmpt)
            self.__iter__()
            yield (np.array(batch_input), batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class generator_nifti():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        return [nib.load(path).get_data() for path in batch]
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)
        return batch[:, :, :, :, :self.channels-1], np.expand_dims(batch[:, :, :, :, self.channels-1], axis=4)
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class generator_matrix_classification():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        return [np.load(path, allow_pickle=True) for path in batch]
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)
        return batch[:, :, :, :, :self.channels-1], np.expand_dims(batch[:, :, :, :, self.channels-1], axis=4)
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class generator_matrix_regression():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]-1
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        return [np.load(path, allow_pickle=True) for path in batch]

    def __gettarget__(self):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, self.channels]
        return batch
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)
        return batch[:, :, :, :, :self.channels], np.expand_dims(self.__gettarget__(), axis=4)
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class generator_mri_regression():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]-1
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        return [nib.load(path).get_data() for path in batch]

    def __gettarget__(self):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, self.channels]
        return batch
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)
        return batch[:, :, :, :, :self.channels], np.expand_dims(self.__gettarget__(), axis=4)
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class generator_mri_segmentation():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]-1
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        if self.shuffle :
            self.shuffle_list()
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        return [nib.load(path).get_data() for path in batch]

    def __gettarget__(self):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, self.channels]
        return[nib.load(path).get_data() for path in batch]
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)
        return batch[:, :, :, :, :self.channels], np.expand_dims(self.__gettarget__(), axis=4)
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len
