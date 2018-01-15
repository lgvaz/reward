from collections import OrderedDict

from scipy.signal import lfilter


def get_obj(config):
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj


def discounted_sum_rewards(rewards, gamma=0.99):
    # Copy needed because torch doesn't support negative strides
    return lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()


def auto_input_shape(obj_config, input_shape):
    name = obj_config['func'].__name__

    if 'FlattenLinear' in name:
        obj_config['in_features'] = input_shape

    if 'ActionLinear' in name:
        obj_config['in_features'] = input_shape

    elif 'Linear' in name:
        assert len(input_shape) == 1, 'Input rank for Linear must be one,'
        'for higher ranks inputs consider using FlattenLinear'
        obj_config['in_features'] = input_shape[0]

    elif 'Conv2d' in name:
        obj_config['in_channels'] = input_shape[0]

    else:
        raise ValueError('Auto input for {} not supported'.format(name))


def get_module_dict(config, input_shape, action_shape):
    # Prepare for creating body network dict
    module_dict = OrderedDict()

    for i, (key, obj_config) in enumerate(config.items()):
        # Calculate the input shape for the first layer
        if i == 0:
            auto_input_shape(obj_config, input_shape)
        # An `Action` layer has the output shape equals to the action shape
        if 'ActionLinear' in obj_config['func'].__name__:
            obj_config['out_features'] = action_shape

        module_dict[key] = get_obj(obj_config)

    return module_dict
