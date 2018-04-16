from collections import defaultdict


class SimpleMemory(defaultdict):
    '''
    Basically a defaultdict with a default list.
    The dict keys can be accessed as attributes.
    '''

    def __init__(self):
        super().__init__(list)

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)
