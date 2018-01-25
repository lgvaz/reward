from collections import OrderedDict
from torchrl.utils import get_obj, Config
from torchrl.nn import SequentialExtended


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


def get_module_list(config, input_shape, action_shape):
    module_list = []
    for i, obj_config in enumerate(config):
        # Calculate the input shape for the first layer
        if i == 0:
            auto_input_shape(obj_config, input_shape)
        # An `Action` layer has the output shape equals to the action shape
        if 'ActionLinear' in obj_config['func'].__name__:
            obj_config['out_features'] = action_shape

        module_list.append(get_obj(obj_config))

    return module_list


def get_module_list(config, input_shape, action_info):
    module_list = []
    for i, obj_config in enumerate(config):
        # Calculate the input shape for the first layer
        if i == 0:
            auto_input_shape(obj_config, input_shape)
        # An `Action` layer has the output shape equals to the action shape
        if 'ActionLinear' in obj_config['func'].__name__:
            obj_config['action_info'] = action_info

        module_list.append(get_obj(obj_config))

    return module_list


def nn_from_config(config, state_info, action_info, body=None, head=None):
    if body is None:
        body_list = get_module_list(
            config=config.body, input_shape=state_info['shape'], action_info=action_info)
        body = SequentialExtended(*body_list)

    if head is None:
        head_list = get_module_list(
            config=config.head,
            input_shape=body.get_output_shape(state_info['shape']),
            action_info=action_info)
        head = SequentialExtended(*head_list)

    return Config(body=body, head=head)
