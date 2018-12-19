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
        img = np.array([cv2.resize(o, self.sz[::-1], interpolation=cv2.INTER_AREA) for o in x])
        return img.reshape((x.shape[0], *self.sz, x.shape[3]))

class Stack(Transform):   
    priority = 1
    def __init__(self, n):
        self.n, self.deque = n, deque(maxlen=n)

    def get(self): return LazyStack(list(self.deque))
        
    def apply(self, x):
        if x.shape[-1] != 1: raise ValueError(f'Can only stack grayscale images (last dim = 1), got {x.shape}')
        if len(self.deque) == 0:
            for _ in range(self.n-1): self.deque.append(x)
        self.deque.append(x)
        return self.get()

class LazyStack:
    def __init__(self, arr): self.arr = arr        
    def __array__(self): return np.array(self.arr).transpose((4, 1, 2, 3, 0))[0]

    @staticmethod
    def from_lists(x): return LazyStackList(x=x)

class LazyStackList:
    def __init__(self, x): self.x = x
    def __array__(self): return np.array([o.arr for o in self.x]).transpose((5, 0, 2, 3, 4, 1))[0]