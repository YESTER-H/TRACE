# SPDX-License-Identifier: Apache-2.0
"""U14 mean blending adapted to RSL-RL5's separate actor/critic models."""
import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories


class U14ProposalActor(MLPModel):
    """508-feature experts, scalar oracle w_T, one native shared Gaussian.

    The router is stored in TensorDict/storage but is never concatenated to the
    network input. RSL-RL5 owns distribution log-prob/entropy/KL, PPO and storage.
    Recovery-only uses w_T=0; Tracking module is retained for architecture audit.
    """

    ALLOW_OBS_NORMALIZATION = False
    STD_TYPES = ("log",)

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512,256,128),
                 activation="elu", obs_normalization=False, distribution_cfg=None):
        if output_dim != 23 or obs["policy"].shape[-1] != 508 or obs["router"].shape[-1] != 1:
            raise ValueError("TRACE requires policy508/router1/action23")
        if obs_groups[obs_set] != ["policy"] or (obs_normalization and not self.ALLOW_OBS_NORMALIZATION):
            raise ValueError("Router must be separate; retain unnormalized U1.5 observation contract")
        cfg = dict(distribution_cfg or {})
        if cfg.get("class_name") != "GaussianDistribution" or cfg.get("std_type") not in self.STD_TYPES:
            raise ValueError(f"One state-independent native {self.STD_TYPES} Gaussian is required")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, cfg)
        # Reuse the already constructed actor as one expert, not a third unused MLP.
        recovery = self.mlp
        del self.mlp
        self.actor_experts = nn.ModuleDict({
            "tracking": MLP(self.obs_dim, output_dim, hidden_dims, activation),
            "recovery": recovery,
        })

    @property
    def log_std(self):
        return self.distribution.log_std_param

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs)
        weight = obs["router"].squeeze(-1)
        if not torch.all(torch.isfinite(weight) & (weight >= 0) & (weight <= 1)):
            raise ValueError("Oracle weight must be finite and in [0,1]")
        mean = latent.new_zeros((len(latent), 23))
        for name, contribution in (("tracking", weight), ("recovery", 1-weight)):
            ids = torch.where(contribution > 0)[0]
            if len(ids):
                mean = mean.index_add(0, ids, self.actor_experts[name](latent[ids])*contribution[ids,None])
        self.distribution.update(mean)
        return self.distribution.sample() if stochastic_output else mean

    def as_jit(self):
        raise NotImplementedError("Oracle router requires a two-input export; do not export as a single-input MLP")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Oracle router requires a two-input export; do not silently omit it")


class PaperAlignedRecoveryActor(U14ProposalActor):
    """U14 actor with StableMimic Table V settings: input normalization, RSL-RL "scalar" std.

    The router stays outside the normalized input (RSL-RL5 normalizes only the
    active "policy" group). Experts, routing and w_T=0 are unchanged.
    """

    ALLOW_OBS_NORMALIZATION = True
    STD_TYPES = ("scalar",)

    @property
    def log_std(self):
        return torch.log(self.distribution.std_param)


class RecoveryMotionExpertActor(MLPModel):
    """One 508->512->256->128->23 expert per get-up reference, oracle one-hot routing.

    Diagnostic split of the U14 Recovery expert: each expert is identical to it,
    the observation/critic/reward/PPO contract is unchanged, and only the
    expert that owns a sample receives its gradient. The [Supine, Prone] router
    is training-only (never concatenated to the input); deployment would need a
    learned proprioceptive gate. The shared log_std[23] is kept.
    """

    EXPERTS = ("supine", "prone")

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512,256,128),
                 activation="elu", obs_normalization=False, distribution_cfg=None):
        if output_dim != 23 or obs["policy"].shape[-1] != 508 or obs["router"].shape[-1] != len(self.EXPERTS):
            raise ValueError("Motion experts require policy508/router2/action23")
        if obs_groups[obs_set] != ["policy"] or obs_normalization:
            raise ValueError("Router must be separate; retain unnormalized U1.5 observation contract")
        cfg = dict(distribution_cfg or {})
        if cfg.get("class_name") != "GaussianDistribution" or cfg.get("std_type") != "log":
            raise ValueError("One state-independent native log-std Gaussian is required")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, cfg)
        supine = self.mlp
        del self.mlp
        self.actor_experts = nn.ModuleDict({
            "supine": supine,
            "prone": MLP(self.obs_dim, output_dim, hidden_dims, activation),
        })

    @property
    def log_std(self):
        return self.distribution.log_std_param

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs)
        router = obs["router"]
        if not torch.all((router == 0) | (router == 1)) or not torch.all(router.sum(-1) == 1):
            raise ValueError("Oracle motion router must be one-hot")
        mean = latent.new_zeros((len(latent), 23))
        for i, name in enumerate(self.EXPERTS):
            ids = torch.where(router[:, i] > 0)[0]
            if len(ids):
                mean = mean.index_add(0, ids, self.actor_experts[name](latent[ids]))
        self.distribution.update(mean)
        return self.distribution.sample() if stochastic_output else mean

    def as_jit(self):
        raise NotImplementedError("Oracle router requires a two-input export; do not export as a single-input MLP")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Oracle router requires a two-input export; do not silently omit it")
