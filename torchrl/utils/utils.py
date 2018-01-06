def get_function(config):
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj
