# SPDX-License-Identifier: Apache-2.0
"""Native RSL-RL5 actor/critic/Gaussian/PPO; no source fork or old runner."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class TraceRecoveryRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1  # Safe smoke default: long training requires explicit override.
    save_interval = 1
    clip_actions = 10.
    experiment_name = "trace_newton_recovery_only"
    run_name = "scratch_correctness_only"
    resume = False
    check_for_nan = True
    obs_groups = {"actor":["policy"],"critic":["critic"]}
    actor = RslRlMLPModelCfg(
        class_name="cyclo_lab.manager_based.trace_recovery.policy:U14ProposalActor",
        hidden_dims=[512,256,128],activation="elu",obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.,std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512,256,128],activation="elu",obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.005,
        num_learning_epochs=5,num_mini_batches=4,learning_rate=1.e-3,schedule="adaptive",
        gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.,
    )


@configclass
class TraceRecoveryPaperAlignedRunnerCfg(TraceRecoveryRunnerCfg):
    """StableMimic Table V: actor/critic normalization and RSL-RL scalar std.

    Entropy stays at the baseline 0.005: Table V's 0.05 (2026-10-27_14-44-20 run)
    grew std 1.0 -> 2.9 in 297 iterations at this reward scale and ended in a NaN reward.
    """
    experiment_name = "trace_newton_recovery_paper_aligned"
    run_name = "scratch_paper_aligned"
    actor = RslRlMLPModelCfg(
        class_name="cyclo_lab.manager_based.trace_recovery.policy:PaperAlignedRecoveryActor",
        hidden_dims=[512,256,128],activation="elu",obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.,std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512,256,128],activation="elu",obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.005,
        num_learning_epochs=5,num_mini_batches=4,learning_rate=1.e-3,schedule="adaptive",
        gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.,
    )


@configclass
class TraceIntegratedRunnerCfg(TraceRecoveryPaperAlignedRunnerCfg):
    """Same actor/critic/PPO as paper-aligned Recovery; w_T now routes Tracking too (U1.5)."""
    experiment_name = "trace_newton_integrated_u15"
    run_name = "scratch_integrated_u15"


@configclass
class TraceIntegratedBalancedRunnerCfg(TraceIntegratedRunnerCfg):
    """Same PPO/actor as run 18; only the env's step budget changes."""
    experiment_name = "trace_newton_integrated_u15_balanced"
    run_name = "scratch_integrated_u15_balanced"


@configclass
class TraceRecoveryMotionExpertRunnerCfg(TraceRecoveryRunnerCfg):
    """Identical PPO/critic/std; only the actor splits Recovery into per-motion experts."""
    experiment_name = "trace_newton_recovery_motion_experts"
    run_name = "scratch_motion_experts"
    actor = RslRlMLPModelCfg(
        class_name="cyclo_lab.manager_based.trace_recovery.policy:RecoveryMotionExpertActor",
        hidden_dims=[512,256,128],activation="elu",obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.,std_type="log"),
    )


@configclass
class TracePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus the StableMimic Table V gate-loss weights."""
    gate_ce_coef: float = .1
    gate_transition_weight: float = 4.
    gate_consistency_coef: float = .01
    gate_alignment_coef: float = .01


@configclass
class TraceIntegratedGatedRunnerCfg(TraceIntegratedBalancedRunnerCfg):
    """Experts as runs 18-20; the oracle w_T is replaced by the learned proprioceptive gate."""
    experiment_name = "trace_newton_integrated_gated"
    run_name = "scratch_integrated_gated"
    actor = RslRlMLPModelCfg(
        class_name="cyclo_lab.manager_based.trace_recovery.gated:TraceGatedActor",
        hidden_dims=[512,256,128],activation="elu",obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.,std_type="scalar"))
    # Entropy stays 0.005 (Table V's 0.05 diverged here, run 16); gate weights are Table V.
    algorithm = TracePpoAlgorithmCfg(
        class_name="cyclo_lab.manager_based.trace_recovery.gated:TracePPO",
        value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.005,
        num_learning_epochs=5,num_mini_batches=4,learning_rate=1.e-3,schedule="adaptive",
        gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.,
    )


@configclass
class TraceIntegratedGatedLookaheadRunnerCfg(TraceIntegratedGatedRunnerCfg):
    experiment_name = "trace_newton_integrated_gated_lookahead"
    run_name = "integrated_gated_lookahead"



@configclass
class TraceIntegratedGatedProneTrimRunnerCfg(TraceIntegratedGatedRunnerCfg):
    experiment_name = "trace_newton_integrated_gated_prone_trim"
    run_name = "integrated_gated_prone_trim"
