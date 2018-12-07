from collections import defaultdict


class SimpleMemory(dict):
    """
    A dict whose keys can be accessed as attributes.

    Parameters
    ----------
    keys: list of strings
        Each key will be initialized as an empty list.
    """

    def __init__(self, *args, keys=None, **kwargs):
        super().__init__(*args, **kwargs)

        keys = keys or []
        for k in keys:
            self[k] = []

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    @classmethod
    def from_dicts(cls, dicts):
        return cls({k: [d[k] for d in dicts] for k in dicts[0]})

    #TODO: Deprecated
    @classmethod
    def from_list_of_dicts(cls, dicts):
        return cls.from_dicts(dicts=dicts)



class DefaultMemory(defaultdict):
    """
    A defaultdict whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(list, *args, **kwargs)

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise AttributeError(key)
