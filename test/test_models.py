import pytest
import torch
import numpy as np
import torchrl as tr
import torch.nn as nn
import torchrl.utils as U
from copy import deepcopy


@pytest.fixture
def replay_batcher():
    env = tr.envs.GymEnv("CartPole-v1")
    runner = tr.runners.SingleRunner(env)
    batcher = tr.batchers.ReplayBatcher(
        runner,
        batch_size=64,
        learning_freq=4,
        replay_buffer_maxlen=1e4,
        init_replays=0.01,
    )
    return batcher


def test_q_model(replay_batcher):
    q_target_fn = lambda batch: 100 * U.to_tensor(np.ones(batch.action.shape))
    nn = tr.arch.MLP.from_batcher(
        batcher=replay_batcher, output_layer=tr.models.QModel.output_layer, hidden=[64]
    )
    q_model = tr.models.QModel(
        nn=nn, batcher=replay_batcher, exploration_rate=0.1, q_target=q_target_fn
    )
    opt = tr.optimizers.SingleOpt(
        model=q_model, opt_fn=torch.optim.SGD, opt_params=dict(lr=1e-2)
    )

    batch = replay_batcher.get_batch(
        lambda state, step: q_model.select_action(state=state, step=step)
    )
    batch = batch.concat_batch()

    for i_action in range(replay_batcher.get_action_info().shape):
        batch.action = i_action * np.ones_like(batch.action)
        batch = batch.apply_to_all(U.to_tensor)

        pred_before = U.to_np(q_model(batch.state_t).mean(0))
        for i in range(1000):
            opt.learn_from_batch(batch=batch, step=i)
        pred_after = U.to_np(q_model(batch.state_t).mean(0))

        expected = pred_before.copy()
        expected[i_action] = 100

        # Tolerance sometimes fail
        np.testing.assert_allclose(pred_after, expected, rtol=.1, atol=1)


@pytest.mark.skip(
    reason="need to decide where to calculate target value (agent or model)"
)
def test_dqn_target_nn_grad(replay_batcher):
    """
    Test if the weights of the target network are not being changed when doing grad descent.
    """
    nn = tr.arch.MLP.from_batcher(
        batcher=replay_batcher, output_layer=tr.models.DQNModel.output_layer
    )
    q_model = tr.models.DQNModel(
        nn=nn, batcher=replay_batcher, exploration_rate=0.1, target_up_freq=5
    )
    opt = tr.optimizers.SingleOpt(
        model=q_model, opt_fn=torch.optim.SGD, opt_params=dict(lr=1e-2)
    )

    for p1, p2 in zip(q_model.nn.parameters(), q_model.target_nn.parameters()):
        assert (p1 == p2).all(), "Parameters should be the same"

    old_target = deepcopy(q_model.target_nn)
    # Perform optimization
    batch = replay_batcher.get_batch(
        lambda state, step: q_model.select_action(state=state, step=step)
    )
    batch = batch.concat_batch()
    opt.learn_from_batch(batch=batch, step=1)

    for p1, p2, p3 in zip(
        q_model.nn.parameters(), q_model.target_nn.parameters(), old_target.parameters()
    ):
        assert (p1 != p2).all(), "Only parameters of the main model should be modified"
        assert (p2 == p3).all(), "Target net parameters should not be modified"

    q_model.update_target_nn(weight=1.)
    for p1, p2 in zip(q_model.nn.parameters(), q_model.target_nn.parameters()):
        assert (p1 == p2).all(), "Parameters should be the same"


@pytest.mark.skip(
    reason="need to decide where to calculate target value (agent or model)"
)
def test_dqn_target_nn_update():
    """
    Tests if the weights are being correctly copied between networks.
    Tests both hard and soft updates
    """
    layer = nn.Linear(4, 2)
    for i, par in enumerate(layer.parameters()):
        par.data.fill_(0)

    model = tr.models.DQNModel(
        nn=layer, batcher=None, exploration_rate=0, target_up_freq=None
    )

    # Change the weights values
    for i, par in enumerate(layer.parameters()):
        par.data.fill_(i)

    # Do a soft update
    model.update_target_nn(weight=0.2)
    for i, par in enumerate(model.target_nn.parameters()):
        assert (par == 0.2 * i).all()

    # Do a hard update
    model.update_target_nn(weight=1.)
    for i, par in enumerate(model.target_net.parameters()):
        assert (par == 1 * i).all()
