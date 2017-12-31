import json


class Config:
    '''
    Configuration file used for initializing an Agent.

    Parameters
    ----------
    configs: Keyword arguments
        Additional parameters that will be stored test

    Returns
    -------
    Config object
        An object containing all configuration details (with possibly nested Config)
    '''

    def __init__(self, **configs):
        for key, value in configs.items():
            self._nested_loader(key, value)

    def __str__(self):
        return (json.dumps(self.__dict__, cls=NestedEncoder, indent=4))

    def _nested_loader(self, key, value):
        if isinstance(value, dict):
            return self.new_section(key, **value)
        else:
            self.__dict__.update({key: value})

    def as_dict(self):
        return self.__dict__

    def new_section(self, name, **configs):
        self._nested_loader(name, Config(**configs))

    def to_json(self, file_path):
        '''
        Saves current configuration to a JSON file.
        The configuration is stored as a nested dictionary

        :ivar file_path: Path to write the file
        '''

        with open(file_path + '.json', 'w') as f:
            json.dump(self, f, cls=NestedEncoder, indent=4)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path + '.json', 'r') as f:
            configs = json.load(f)
        return cls(**configs)

    @classmethod
    def from_default(cls, name):
        if name == 'PPO':
            return cls.from_json('ppo5')


class NestedEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'as_dict'):
            return obj.as_dict()
        else:
            return json.JSONEncoder.default(self, obj)
