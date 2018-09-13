import pdb
import torch
from torchrl.agents import BaseAgent


# TODO: Q_model2, copy or instantiate again?
class SAC(BaseAgent):
    def __init__(
        self,
        batcher,
        optimizer,
        action_fn,
        *,
        policy_model,
        q_model,
        v_model,
        log_dir=None,
        **kwargs
    ):
        super().__init__(
            batcher=batcher,
            optimizer=optimizer,
            action_fn=action_fn,
            log_dir=log_dir,
            **kwargs
        )
        self.register_model("policy", policy_model)
        self.register_model("q", q_model)
        self.register_model("v", v_model)
        # self.policy_model = policy_model
        # self.q_model = q_model
        # self.v_model = v_model
        self.p_opt = torch.optim.Adam(self.models.policy.nn.parameters(), lr=3e-4)
        self.q_opt = torch.optim.Adam(self.models.q.nn.parameters(), lr=3e-4)
        self.v_opt = torch.optim.Adam(self.models.v.nn.parameters(), lr=3e-4)

    def step(self):
        batch = self.generate_batch()
        # TODO: Check shapes
        batch = batch.concat_batch()

        # Policy
        dist = self.models.policy.create_dist(state=batch.state_t)
        # action, pre_activation = dist.rsample_with_pre()
        action, pre_activation = dist.sample_with_pre()
        log_prob = dist.log_prob(action).sum(-1)
        # Q
        q_t_replay_act = self.models.q((batch.state_t, batch.action))
        q_t_new_act = self.models.q((batch.state_t, action))
        # V
        v_t = self.models.v(batch.state_t)
        v_target_tp1 = self.models.v.forward_target(batch.state_tp1)

        # Q target calculation
        with torch.no_grad():
            batch.qtarget = (
                batch.reward + (1 - batch.done) * 0.99 * v_target_tp1.squeeze()
            )

        # V target calculation
        with torch.no_grad():
            batch.vtarget = q_t_new_act.squeeze() - log_prob

        # Policy
        batch.q_t_new_act = q_t_new_act
        batch.v_t = v_t
        # Alternatively just pass dist and new_action
        batch.new_action = action
        batch.new_pre_activation = pre_activation
        batch.new_dist = dist
        batch.log_prob = log_prob
        # batch.dist_mean = dist.loc
        # batch.dist_log_std = dist.scale.log()
        # pdb.set_trace()

        # self.train_models(batch)
        # p_loss = (log_prob.squeeze() - q_t_new_act.squeeze()).mean()
        log_prob_target = q_t_new_act.squeeze() - v_t.squeeze()
        p_loss = (
            log_prob.squeeze()
            * (log_prob.squeeze() - log_prob_target.squeeze()).detach()
        ).mean()

        mean_loss = 1e-3 * dist.loc.pow(2).mean()
        std_loss = 1e-3 * dist.scale.log().pow(2).mean()
        p_loss += mean_loss + std_loss

        v_loss = ((v_t.squeeze() - batch.vtarget.squeeze().detach()).pow(2)).mean()
        q_loss = (
            (q_t_replay_act.squeeze() - batch.qtarget.squeeze().detach()).pow(2)
        ).mean()

        self.p_opt.zero_grad()
        p_loss.backward()
        p_grad = torch.nn.utils.clip_grad_norm_(
            self.models.policy.parameters(), float("inf")
        )
        self.p_opt.step()

        self.v_opt.zero_grad()
        v_loss.backward()
        v_grad = torch.nn.utils.clip_grad_norm_(
            self.models.v.parameters(), float("inf")
        )
        self.v_opt.step()

        self.q_opt.zero_grad()
        q_loss.backward()
        q_grad = torch.nn.utils.clip_grad_norm_(
            self.models.q.parameters(), float("inf")
        )
        self.q_opt.step()

        self.models.v.update_target_nn(0.005)

        if self.num_iters % self.log_freq == 0:
            for model in self.models.values():
                model.write_logs(batch=batch)

            self.models.policy.add_log("loss", p_loss)
            self.models.v.add_log("loss", v_loss)
            self.models.q.add_log("loss", q_loss)
            self.models.policy.add_log("grad_norm", p_grad)
            self.models.v.add_log("grad_norm", v_grad)
            self.models.q.add_log("grad_norm", q_grad)
