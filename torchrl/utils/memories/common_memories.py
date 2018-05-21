from collections import defaultdict


class SimpleMemory(dict):
    '''
    A dict whose keys can be accessed as attributes.
    '''

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)


class DefaultMemory(defaultdict):
    '''
    A defaultdict whose keys can be accessed as attributes.
    '''

    def __init__(self):
        super().__init__(list)

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)
