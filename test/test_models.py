import pytest
import torch
import numpy as np
import torchrl as tr
import torch.nn as nn
import torchrl.utils as U


@pytest.fixture
def nn_config():
    return U.Config(
        nn_config=U.Config(
            body=[dict(func=nn.Linear, out_features=64), dict(func=nn.Tanh)]
        )
    )


@pytest.fixture
def replay_batcher():
    env = tr.envs.GymEnv("CartPole-v1")
    runner = tr.runners.SingleRunner(env)
    batcher = tr.batchers.ReplayBatcher(
        runner,
        batch_size=64,
        steps_per_batch=4,
        replay_buffer_maxlen=1e4,
        init_replays=0.01,
    )
    return batcher


def test_q_model(nn_config, replay_batcher):
    q_model = tr.models.QModel.from_config(
        config=nn_config, batcher=replay_batcher, exploration_rate=0.1
    )
    opt = tr.optimizers.SingleOpt(
        model=q_model, opt_fn=torch.optim.SGD, opt_params=dict(lr=1e-2)
    )

    batch = replay_batcher.get_batch(
        lambda state, step: q_model.select_action(model=q_model, state=state, step=step)
    )
    batch = batch.concat_batch()

    for i_action in range(replay_batcher.get_action_info().shape):
        batch.action = i_action * np.ones_like(batch.action)
        batch.vtarget = 100 * np.ones(batch.action.shape)
        batch = batch.apply_to_all(U.to_tensor)

        pred_before = U.to_np(q_model(batch.state_t).mean(0))
        for i in range(1000):
            opt.learn_from_batch(batch=batch, step=i)
        pred_after = U.to_np(q_model(batch.state_t).mean(0))

        expected = pred_before.copy()
        expected[i_action] = U.to_np(batch.vtarget).mean()

        np.testing.assert_allclose(pred_after, expected, rtol=.1, atol=1)
