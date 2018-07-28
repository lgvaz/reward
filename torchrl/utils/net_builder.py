from collections import OrderedDict
from torchrl.utils import get_obj, Config
from torchrl.nn import SequentialExtended


def auto_input_shape(obj_config, input_shape):
    """
    Create the right input parameter for the type of layer

    Parameters
    ----------
    obj_config: dict
        A dict containing the function and the parameters for creating the object.
    input_shape: list
        The input dimensions.
    """
    name = obj_config["func"].__name__

    if "FlattenLinear" in name:
        obj_config["in_features"] = input_shape

    elif "ActionLinear" in name:
        obj_config["in_features"] = input_shape

    elif "Linear" in name:
        assert len(input_shape) == 1, "Input rank for Linear must be one,"
        "for higher ranks inputs consider using FlattenLinear"
        obj_config["in_features"] = input_shape[0]

    elif "Conv2d" in name:
        obj_config["in_channels"] = input_shape[0]

    else:
        raise ValueError("Auto input for {} not supported".format(name))


def get_module_list(config, input_shape, action_info):
    """
    Receives a config object and creates a list of layers.

    Parameters
    ----------
    config: Config
        The configuration object that should contain the basic network structure.
    input_shape: list
        The input dimensions.
    action_info: dict
        Dict containing information about the environment actions (e.g. shape).

    Returns
    -------
    list of layers
        A list containing all the instantiated layers.
    """
    module_list = []
    for i, obj_config in enumerate(config):
        # Calculate the input shape for the first layer
        if i == 0:
            auto_input_shape(obj_config, input_shape)
        # An `Action` layer has the output shape equals to the action shape
        if "ActionLinear" in obj_config["func"].__name__:
            obj_config["action_info"] = action_info

        module_list.append(get_obj(obj_config))

    return module_list


def nn_from_config(config, state_info, action_info, body=None, head=None):
    """
    Creates a pytorch model following the instructions of config.

    Parameters
    ----------
    config: Config
        The configuration object that should contain the basic network structure.
    state_info: dict
        Dict containing information about the environment states (e.g. shape).
    action_info: dict
        Dict containing information about the environment actions (e.g. shape).
    body: Module
        If given use it instead of creating (Default is None).
    head: Module
        If given use it instead of creating (Default is None).

    Returns
    -------
    torchrl.SequentialExtended
        A torchrl NN (basically a pytorch NN with extended functionalities).
    """
    if body is None:
        body_list = get_module_list(
            config=config.body, input_shape=state_info.shape, action_info=action_info
        )
        body = SequentialExtended(*body_list)

    if head is None:
        head_list = get_module_list(
            config=config.head,
            input_shape=body.get_output_shape(state_info.shape),
            action_info=action_info,
        )
        head = SequentialExtended(*head_list)

    return SequentialExtended(OrderedDict([("body", body), ("head", head)]))
