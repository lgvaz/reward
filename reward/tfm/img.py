import numpy as np
import cv2
from abc import ABC, abstractmethod
from collections import deque


class Transform(ABC):    
    priority = 5
    def __call__(self, x): return self.apply(x=x)
    
    @abstractmethod
    def apply(self, x): pass

class Gray(Transform):
    priority = 8
    def apply(self, x): return np.dot(x[..., :3], [0.299, 0.587, 0.114])[..., None].astype(x.dtype)

class Resize(Transform):    
    priority = 7
    def __init__(self, sz):
        if len(sz) != 2: raise ValueError(f'sz should be (x, y), got {sz}')
        self.sz = tuple(sz)

    def apply(self, x): 
        img = np.array([cv2.resize(o, self.sz[::-1], interpolation=cv2.INTER_NEAREST) for o in x])
        return img.reshape((x.shape[0], *self.sz, x.shape[3]))

class Stack(Transform):   
    priority = 1
    def __init__(self, n, in_sz):
        raise NotImplementedError
        self.deque = deque(maxlen=n)
        for _ in range(n): self.deque.append(np.zeros(in_sz, dtype='uint8'))
            
    def get(self): return LazyStack(self.deque)
        
    def apply(self, x):
        self.deque.append(x)
        return self.get()

class LazyStack():
    def __init__(self, arr): self.arr = arr        
    @property
    def shape(self): raise NotImplementedError     
    def to_pil(self):
        arr = np.array([np.array(o) for o in self.arr])
        return PIL.Image.fromarray(np.moveaxis(arr, 0, -1))

