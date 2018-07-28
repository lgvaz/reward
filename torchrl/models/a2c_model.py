from torchrl.models import VanillaPGModel


class A2CModel(VanillaPGModel):
    """
    A2C is just a parallel implementation of the actor-critic algorithm.

    So just be sure to create a list of envs and pass to
    :class:`torchrl.envs.ParallelEnv` to reproduce A2C.
    """
