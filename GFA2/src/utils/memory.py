import h5py
import numpy as np
import os

class HDF5FeatureWriter:
    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.dataset = None
        self.counter = 0
        self._initialized = False
        
    def __enter__(self):
        self.file = h5py.File(self.path, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def create_dataset(self, feat_dim: int):
        self.dataset = self.file.create_dataset(
            "features",
            shape=(0, feat_dim),
            maxshape=(None, feat_dim),
            chunks=(1000, feat_dim),
            dtype=np.float32
        )
        
    def write_batch(self, features: np.ndarray):
        if features.size == 0:
            return
            
        if not self._initialized:
            self.create_dataset(features.shape[1])
            self._initialized = True
            
        batch_size = features.shape[0]
        self.dataset.resize(self.counter + batch_size, axis=0)
        self.dataset[self.counter:self.counter+batch_size] = features
        self.counter += batch_size
        
    def close(self):
        if self.file is not None:
            self.file.close()
            print(f"Feature file saved to: {os.path.abspath(self.path)}")