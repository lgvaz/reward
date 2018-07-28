from collections import defaultdict


class SimpleMemory(dict):
    """
    A dict whose keys can be accessed as attributes.

    Parameters
    ----------
    initial_keys: list of strings
        Each key will be initialized as an empty list.
    """

    def __init__(self, *args, initial_keys=None, **kwargs):
        super().__init__(*args, **kwargs)

        initial_keys = initial_keys or []
        for k in initial_keys:
            self[k] = []

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)

    @classmethod
    def from_list_of_dicts(cls, dicts):
        return cls({k: [d[k] for d in dicts] for k in dicts[0]})


class DefaultMemory(defaultdict):
    """
    A defaultdict whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(list, *args, **kwargs)

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)
