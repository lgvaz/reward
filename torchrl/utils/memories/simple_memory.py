class SimpleMemory(dict):
    '''
    A dict whose keys can be accessed as attributes.
    '''

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)
