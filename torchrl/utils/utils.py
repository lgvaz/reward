import copy


def get_function(config):
    config = copy.deepcopy(config)
    func = eval(config.pop('func'))

    return func(**config)
