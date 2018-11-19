import PIL
import numpy, torchvision.transforms.functional as np, ttfm
from abc import ABC, abstractmethod
from collections import deque


class Transform(ABC):    
    priority = 5
    def __call__(self, x): return self.apply(x=x)
    
    @abstractmethod
    def apply(self, x): pass

class Gray(Transform):
    priority = 8
    def apply(self, x): return ttfm.to_grayscale(x)

class Resize(Transform):    
    priority = 7
    def __init__(self, sz): self.sz = sz     
    def apply(self, x): return ttfm.resize(x, size=self.sz)

class Stack(Transform):   
    priority = 1
    def __init__(self, n, in_sz):
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

