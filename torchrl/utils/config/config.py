import yaml
from collections import OrderedDict


class Config:
    '''
    Configuration object used for initializing an Agent.
    It maintains the order from which the attributes have been set.

    Parameters
    ----------
    configs: Keyword arguments
        Additional parameters that will be stored.

    Returns
    -------
    Config object
        An object containing all configuration details (with possibly nested Config).
    '''

    def __init__(self, **configs):
        # We want to maintain the order of the attributes,
        # this is especially necessary when defining NNs architectures
        self.__dict__['_attrs'] = OrderedDict()
        for key, value in configs.items():
            self._nested_loader(key, value)

    def __getattr__(self, value):
        try:
            return self.__dict__['_attrs'][value]
        except:
            raise AttributeError(value)

    def __setattr__(self, key, value):
        self.__dict__['_attrs'][key] = value

    def __str__(self):
        return yaml.dump(self.as_dict(), default_flow_style=False)

    def _nested_loader(self, key, value):
        if isinstance(value, OrderedDict):
            return self.new_section(key, **value)
        else:
            setattr(self, key, value)

    def pop(self, key):
        return self.__dict__['_attrs'].pop(key)

    def as_dict(self):
        '''
        Returns all object attributes as a nested OrderedDict.

        Returns
        -------
        dict
            Nested OrderedDict containing all object attributes.
        '''
        return self.__dict__['_attrs']

    def new_section(self, name, **configs):
        '''
        Creates a new Config object and add as an attribute of this instance.

        Parameters
        ----------
        name: str
            Name of the new section.
        configs: Keyword arguments
            Parameters that will be stored in this section, accepts nested parameters.

        Examples
        --------
        Simple use case::

            config.new_section('new_section_name', attr1=value1, attr2=value2, ...)

        Nested parameters::

            config.new_section('new_section_name', attr1=Config(attr1=value1, attr2=value2))

        It's possible to access the variable like so::

            config.new_section_name.attr1
        '''
        self._nested_loader(name, Config(**configs))

    def save(self, file_path):
        '''
        Saves current configuration to a JSON file.
        The configuration is stored as a nested dictionary (maintaining the order).

        Parameters
        ----------
        file_path: str
            Path to write the file
        '''
        with open(file_path + '.yaml', 'w') as f:
            yaml.dump(self, f, default_flow_style=False)

    @staticmethod
    def load(file_path):
        '''
        Loads configuration from a JSON file.

        Parameters
        ----------
        file_path: str
            Path of the file to be loaded.

        Returns
        -------
        Config
            A configuration object loaded from a JSON file
        '''
        with open(file_path + '.yaml', 'r') as f:
            return yaml.load(f)

    @classmethod
    def from_default(cls, name):
        '''
        Loads configuration from a default agent.

        Parameters
        ----------
        name: str
            Name of the desired config file ('VanillaPG', add_more)

        Returns
        -------
        Config
            A configuration object loaded from a JSON file
        '''
        if name == 'PPO':
            return cls.load('CHANGE')
