# SPDX-License-Identifier: Apache-2.0
"""StableMimic-style proprioceptive gate: two experts blended by a learned gate.

    mu_T = f_T(o_pi), mu_R = f_R(o_pi)                (experts: 508 -> 512-256-128 -> 23)
    [w_T, w_R] = softmax(f_G(H4(o_prop)) / tau)      (gate: 4x75 -> 512-64 -> 2)
    mu = w_T mu_T + w_R mu_R                          (one shared scalar-std Gaussian)

The gate sees proprioception only (angular velocity, joint position/velocity, previous
action, projected gravity), never the command or any Recovery identity/frame. The
oracle regime is training-only supervision ("gate_target" group: [y_T, same regime as
the previous step, Transition]). PPO gradients flow through the blend; TracePPO adds

    L = L_PPO + c_CE L_CE + c_cons L_cons + c_align L_align
    L_CE    = E[ w_t * CE(y, w) ],  w_t = transition weight in Transition, else 1
    L_cons  = E[ same * ||w_t - w_{t-1}||^2 ]   (w_{t-1}: gate on the previous H4 window)
    L_align = E[ transition * ||mu_T - mu_R||^2 ]

The "gate" observation group keeps five frames per term so that both H4 windows (t and
t-1) are available from one stored sample; the deployable gate uses only the newest four.
"""
import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories

GATE_TERM_DIMS = (3, 23, 23, 23, 3)  # base_ang_vel, joint_pos, joint_vel, last_action, projected_gravity
GATE_FRAME_DIM = sum(GATE_TERM_DIMS)
GATE_HISTORY = 5
GATE_WINDOW = 4


def gate_frames(gate_obs):
    """Term-wise flattened [N, sum(H*d)] (oldest frame first per term) -> [N, H, 75]."""
    terms = torch.split(gate_obs, [GATE_HISTORY*d for d in GATE_TERM_DIMS], dim=-1)
    return torch.cat([t.reshape(len(gate_obs), GATE_HISTORY, d) for t, d in zip(terms, GATE_TERM_DIMS)], -1)


class TraceGatedActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128),
                 activation="elu", obs_normalization=True, distribution_cfg=None,
                 gate_hidden_dims=(512, 64), gate_temperature=1.):
        if output_dim != 23 or obs["policy"].shape[-1] != 508:
            raise ValueError("TRACE gated actor requires policy508/action23")
        if obs["gate"].shape[-1] != GATE_HISTORY*GATE_FRAME_DIM or obs["gate_target"].shape[-1] != 3:
            raise ValueError("Gate requires gate375 (5 frames x 75) and gate_target3")
        if obs_groups[obs_set] != ["policy"]:
            raise ValueError("Experts read only the policy group; gate groups stay separate")
        cfg = dict(distribution_cfg or {})
        if cfg.get("class_name") != "GaussianDistribution" or cfg.get("std_type") != "scalar":
            raise ValueError("One state-independent scalar-std Gaussian is required")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, cfg)
        recovery = self.mlp
        del self.mlp
        self.actor_experts = nn.ModuleDict({
            "tracking": MLP(self.obs_dim, output_dim, hidden_dims, activation),
            "recovery": recovery,
        })
        self.gate = MLP(GATE_WINDOW*GATE_FRAME_DIM, 2, list(gate_hidden_dims), activation)
        self.gate_normalizer = EmpiricalNormalization(GATE_FRAME_DIM)
        self.gate_temperature = gate_temperature
        self.last_gate = None  # tensors of the latest forward, read by TracePPO and diagnostics

    @property
    def log_std(self):
        return torch.log(self.distribution.std_param)

    def gate_logits(self, frames):
        """frames [N, 4, 75] (raw) -> logits [N, 2] for [Tracking, Recovery]."""
        return self.gate(self.gate_normalizer(frames).flatten(1))/self.gate_temperature

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs)
        mu_t = self.actor_experts["tracking"](latent)
        mu_r = self.actor_experts["recovery"](latent)
        frames = gate_frames(obs["gate"])
        logits = self.gate_logits(frames[:, -GATE_WINDOW:])
        weights = torch.softmax(logits, -1)
        mean = weights[:, :1]*mu_t+weights[:, 1:]*mu_r
        self.last_gate = {"logits": logits, "weights": weights, "mu_t": mu_t, "mu_r": mu_r, "frames": frames}
        self.distribution.update(mean)
        return self.distribution.sample() if stochastic_output else mean

    def update_normalization(self, obs):
        super().update_normalization(obs)
        if self.obs_normalization:
            self.gate_normalizer.update(gate_frames(obs["gate"])[:, -1])

    def auxiliary_losses(self, gate_target, transition_weight):
        """Gate CE / routing consistency / expert alignment for the latest forward()."""
        g = self.last_gate
        y_t = gate_target[:, 0].clamp(0, 1)
        same, transition = gate_target[:, 1], gate_target[:, 2]
        target = torch.stack((y_t, 1-y_t), -1)
        ce = -(target*torch.log_softmax(g["logits"], -1)).sum(-1)
        ce = (ce*torch.where(transition > 0, transition_weight, 1.)).mean()
        previous = torch.softmax(self.gate_logits(g["frames"][:, :GATE_WINDOW]), -1)
        consistency = (same*(g["weights"]-previous).square().sum(-1)).mean()
        alignment = (transition*(g["mu_t"]-g["mu_r"]).square().sum(-1)).mean()
        return ce, consistency, alignment

    def as_jit(self):
        raise NotImplementedError("Gated actor needs a policy+gate two-input export")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Gated actor needs a policy+gate two-input export")


class TracePPO(PPO):
    """RSL-RL5 PPO.update() with the StableMimic gate losses added to the same backward.

    Only the configuration TRACE uses is supported: feed-forward actor/critic, no RND,
    no symmetry. Everything else is the library update, line for line.
    """

    def __init__(self, actor, critic, storage, gate_ce_coef=.1, gate_transition_weight=4.,
                 gate_consistency_coef=.01, gate_alignment_coef=.01, **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd or self.symmetry or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("TracePPO supports feed-forward actor/critic without RND/symmetry")
        self.gate_ce_coef = gate_ce_coef
        self.gate_transition_weight = gate_transition_weight
        self.gate_consistency_coef = gate_consistency_coef
        self.gate_alignment_coef = gate_alignment_coef

    def update(self):
        sums = dict.fromkeys(("value", "surrogate", "entropy", "gate_ce", "gate_consistency", "gate_alignment"), 0.)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages-batch.advantages.mean())/(batch.advantages.std()+1e-8)
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                       stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params))
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl*2.:
                            self.learning_rate = max(1e-5, self.learning_rate/1.5)
                        elif kl_mean < self.desired_kl/2. and kl_mean > 0.:
                            self.learning_rate = min(1e-2, self.learning_rate*1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob-torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages)*ratio
            surrogate_clipped = -torch.squeeze(batch.advantages)*torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values+(values-batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values-batch.returns).pow(2), (value_clipped-batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns-values).pow(2).mean()
            ce, consistency, alignment = self.actor.auxiliary_losses(
                batch.observations["gate_target"], self.gate_transition_weight)
            loss = (surrogate_loss+self.value_loss_coef*value_loss-self.entropy_coef*entropy.mean()
                    +self.gate_ce_coef*ce+self.gate_consistency_coef*consistency+self.gate_alignment_coef*alignment)

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            for key, value in (("value", value_loss), ("surrogate", surrogate_loss), ("entropy", entropy.mean()),
                               ("gate_ce", ce), ("gate_consistency", consistency), ("gate_alignment", alignment)):
                sums[key] += value.item()
        self.storage.clear()
        return {key: value/(self.num_learning_epochs*self.num_mini_batches) for key, value in sums.items()}
