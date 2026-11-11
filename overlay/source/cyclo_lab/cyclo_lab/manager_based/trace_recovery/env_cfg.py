# SPDX-License-Identifier: Apache-2.0
"""Additive TRACE Recovery-only task; no upstream mimic/asset modification."""
import copy
import os
from dataclasses import fields
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab_newton.sensors.contact_sensor import ContactSensorCfg as ShapeContactSensorCfg
from cyclo_lab.assets.robots.K1_rev1 import K1_REV1_URDF_PATH
from cyclo_lab.manager_based.mimic.config.k1_rev1.base_env_cfg import K1Rev1MimicEnvCfg, TRACKED_BODY_NAMES
from .commands import RecoveryOnlyCommandCfg
from .body_adapter import JOINT_NAMES
from . import mdp
from . import integrated


@configclass
class OracleObservationCfg(ObservationGroupCfg):
    weight = ObservationTermCfg(func=mdp.oracle_weight)
    concatenate_terms = True
    enable_corruption = False


@configclass
class TraceRecoveryEnvCfg(K1Rev1MimicEnvCfg):
    """Only Recovery action/reward, two hidden references sampled 50/50."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.episode_length_s = 30.
        self.export_sim2real_cfg = False
        inputs = os.environ.get("TRACE_MOTION_INPUT_DIR", os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../../data/motions/K1_rev1/trace_inputs")))
        self.commands.reference_trajectory = RecoveryOnlyCommandCfg(
            trajectory_file=os.path.join(inputs,"tracking.npz"),
            supine_trajectory_file=os.path.join(inputs,"supine.npz"),
            prone_trajectory_file=os.path.join(inputs,"prone.npz"),
            urdf_path=K1_REV1_URDF_PATH, body_names=TRACKED_BODY_NAMES,
        )
        self.actions.joint_pos.joint_names = list(JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.events.joint_home_offset_noise.func = mdp.canonical_home_joint_offset_noise
        policy = self.observations.policy
        policy.motion_command.func = mdp.live_tracking_command
        policy.motion_anchor_ori_b.func = mdp.live_anchor_orientation
        policy.joint_pos_rel.func = mdp.joint_position
        policy.joint_vel_rel.func = mdp.joint_velocity
        # U1.5 really stored term-wise H4, gravity appended after previous action.
        policy.projected_gravity = ObservationTermCfg(
            func=base_mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-.05,n_max=.05))
        policy.history_length = 4
        policy.flatten_history_dim = True
        self.observations.critic.joint_pos.func = mdp.joint_position
        self.observations.critic.joint_vel.func = mdp.joint_velocity
        self.observations.critic.base_lin_vel.func = mdp.base_linear_velocity
        self.observations.router = OracleObservationCfg()
        self.scene.contact_forces.debug_vis = False
        self.scene.head_contact_forces = ShapeContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            sensor_shape_prim_expr=["{ENV_REGEX_NS}/Robot/Geometry/pelvis/torso_link/head_link_1/head_link"],
            history_length=3, debug_vis=False,
        )
        self.rewards.reference_anchor_position.func = mdp.root_height_reward
        for name in ("reference_anchor_orientation","reference_body_position","reference_body_orientation",
                     "reference_body_linear_velocity","reference_body_angular_velocity"):
            term = getattr(self.rewards,name)
            term.params["kernel_name"] = term.func.__name__
            term.func = mdp.imitation_reward
        self.rewards.undesired_contacts.func = mdp.undesired_contacts
        self.rewards.undesired_contacts.params = {"threshold":1.}
        self.terminations.anchor_height = None
        self.terminations.anchor_ori = None
        self.terminations.end_body_pos = None
        self.terminations.nan_detected.func = mdp.nonfinite_state
        self.terminations.nan_detected.params = {}
        self.terminations.recovery_success = TerminationTermCfg(func=mdp.success_completion,time_out=False)
        self.terminations.recovery_persistent_failure = TerminationTermCfg(func=mdp.persistent_failure,time_out=False)
        self.terminations.recovery_grace_failure = TerminationTermCfg(func=mdp.grace_failure,time_out=False)


@configclass
class TraceRecoveryPlayEnvCfg(TraceRecoveryEnvCfg):
    """No recovery perturbation/actor corruption; forced frame0 is test-only."""
    def __post_init__(self):
        super().__post_init__()
        self.commands.reference_trajectory.reset_perturbations = False
        self.observations.policy.enable_corruption = False


@configclass
class RecoveryIdentityObservationCfg(ObservationGroupCfg):
    identity = ObservationTermCfg(func=mdp.oracle_recovery_identity)
    concatenate_terms = True
    enable_corruption = False


@configclass
class TraceRecoveryMotionExpertEnvCfg(TraceRecoveryEnvCfg):
    """Same MDP; only the training-only router becomes a [Supine, Prone] one-hot."""
    def __post_init__(self):
        super().__post_init__()
        self.observations.router = RecoveryIdentityObservationCfg()


@configclass
class TraceRecoveryMotionExpertPlayEnvCfg(TraceRecoveryPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.router = RecoveryIdentityObservationCfg()


# StableMimic Appendix Table V reset ranges (joint-velocity noise is unspecified there; kept).
PAPER_RESET_POSE_NOISE = {"x":(-.05,.05),"y":(-.05,.05),"z":(-.01,.01),
                          "roll":(-.1,.1),"pitch":(-.1,.1),"yaw":(-.2,.2)}
PAPER_RESET_VELOCITY_NOISE = {"x":(-.5,.5),"y":(-.5,.5),"z":(-.2,.2),
                              "roll":(-.52,.52),"pitch":(-.52,.52),"yaw":(-.78,.78)}
PAPER_RESET_JOINT_POSITION_NOISE = .1
# Terminal completion bonus. Rewards are scaled by step_dt (0.02 s), so this is a
# one-time return of 10: ~50 steps at the observed mean Recovery reward (~0.2/step),
# 3-4x the whole 2 s stall window, ~23% of the baseline mean episode return (43.7).
SUCCESS_BONUS_WEIGHT = 500.


def _configure_paper_aligned(cfg):
    command = cfg.commands.reference_trajectory
    command.reset_pose_noise = dict(PAPER_RESET_POSE_NOISE)
    command.reset_velocity_noise = dict(PAPER_RESET_VELOCITY_NOISE)
    command.reset_joint_position_noise = PAPER_RESET_JOINT_POSITION_NOISE
    cfg.rewards.recovery_success = RewardTermCfg(func=mdp.success_bonus, weight=SUCCESS_BONUS_WEIGHT)


@configclass
class TraceRecoveryPaperAlignedEnvCfg(TraceRecoveryEnvCfg):
    """Unified Recovery-only MDP plus Table V reset ranges and a terminal success bonus."""
    def __post_init__(self):
        super().__post_init__()
        _configure_paper_aligned(self)


@configclass
class TraceRecoveryPaperAlignedPlayEnvCfg(TraceRecoveryPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_paper_aligned(self)


@configclass
class IntegratedRouterObservationCfg(ObservationGroupCfg):
    weight = ObservationTermCfg(func=integrated.oracle_tracking_weight)
    concatenate_terms = True
    enable_corruption = False


def _configure_integrated(cfg):
    old = cfg.commands.reference_trajectory
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "class_type"}
    cfg.commands.reference_trajectory = integrated.TraceIntegratedCommandCfg(**values)
    cfg.observations.router = IntegratedRouterObservationCfg()
    cfg.rewards.reference_anchor_position.func = integrated.integrated_imitation_reward
    cfg.rewards.reference_anchor_position.params["kernel_name"] = "reference_anchor_position_tracking"
    for name in ("reference_anchor_orientation","reference_body_position","reference_body_orientation",
                 "reference_body_linear_velocity","reference_body_angular_velocity"):
        getattr(cfg.rewards,name).func = integrated.integrated_imitation_reward
    # Completion hands off to Transition instead of ending the episode.
    cfg.terminations.recovery_success = None
    cfg.rewards.recovery_success.func = integrated.handoff_bonus
    cfg.terminations.tracking_failure = TerminationTermCfg(func=integrated.tracking_failure,time_out=False)
    # Appended LAST: every failure/safety/time-out above takes precedence (U1.5).
    cfg.terminations.post_tracking_horizon_complete = TerminationTermCfg(
        func=integrated.post_tracking_horizon_complete,time_out=True)


@configclass
class TraceIntegratedEnvCfg(TraceRecoveryPaperAlignedEnvCfg):
    """U1.5 Tracking + unified Recovery with the paper-aligned Recovery MDP."""
    def __post_init__(self):
        super().__post_init__()
        _configure_integrated(self)


@configclass
class TraceIntegratedPlayEnvCfg(TraceRecoveryPaperAlignedPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_integrated(self)


def _configure_integrated_balanced(cfg):
    """Recovery gets ~half of the training steps instead of ~13%.

    Tracking-origin episodes stop at 12 s (two Hiphop loops) as a time-out, and resets
    are Tracking 30 / Recovery 70 (Supine 35 / Prone 35). Recovery-origin episodes still
    use the 30 s global limit (they need up to 12.9 s from frame 0).
    """
    cfg.commands.reference_trajectory.reset_regime_probabilities = (.3,.7)
    post = cfg.terminations.post_tracking_horizon_complete
    delattr(cfg.terminations,"post_tracking_horizon_complete")
    cfg.terminations.tracking_origin_horizon_complete = TerminationTermCfg(
        func=integrated.tracking_origin_horizon_complete,params={"horizon_s":12.},time_out=True)
    cfg.terminations.post_tracking_horizon_complete = post  # still evaluated last


@configclass
class TraceIntegratedBalancedEnvCfg(TraceIntegratedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_integrated_balanced(self)


@configclass
class TraceIntegratedBalancedPlayEnvCfg(TraceIntegratedPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_integrated_balanced(self)


# Supine get-up without its first 100 frames (0-99 lie still; joint speed ~0.02 rad/s).
# From frame 0 the run-20 policy stood ~100 frames ahead of the original reference.
SUPINE_TRIM100_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../data/motions/K1_rev1/get_up/supine/supine_getup_trim100.npz"))


@configclass
class GateObservationCfg(ObservationGroupCfg):
    """Proprioception only, five frames per term (gate uses the newest four; t-1 window for consistency).

    Terms are copied from the policy group in _configure_gated so that functions and noise match.
    """
    base_ang_vel = ObservationTermCfg(func=base_mdp.base_ang_vel)
    joint_pos = ObservationTermCfg(func=mdp.joint_position)
    joint_vel = ObservationTermCfg(func=mdp.joint_velocity)
    last_action = ObservationTermCfg(func=base_mdp.last_action)
    projected_gravity = ObservationTermCfg(func=base_mdp.projected_gravity)
    concatenate_terms = True
    history_length = 5
    flatten_history_dim = True


@configclass
class GateTargetObservationCfg(ObservationGroupCfg):
    target = ObservationTermCfg(func=integrated.gate_supervision)
    concatenate_terms = True
    enable_corruption = False


def _configure_gated(cfg):
    """Final TRACE structure: proprioceptive gate, trimmed Supine, early-upright handoff."""
    command = cfg.commands.reference_trajectory
    command.supine_trajectory_file = SUPINE_TRIM100_FILE
    command.early_upright_steps = 25  # 0.5 s upright
    policy = cfg.observations.policy
    gate = GateObservationCfg()
    for name, source in (("base_ang_vel","base_ang_vel"),("joint_pos","joint_pos_rel"),("joint_vel","joint_vel_rel"),
                         ("last_action","last_action"),("projected_gravity","projected_gravity")):
        setattr(gate, name, copy.deepcopy(getattr(policy, source)))
    gate.enable_corruption = policy.enable_corruption
    cfg.observations.gate = gate
    cfg.observations.gate_target = GateTargetObservationCfg()
    cfg.observations.router = None  # the oracle weight is now only a gate label


@configclass
class TraceIntegratedGatedEnvCfg(TraceIntegratedBalancedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_gated(self)


@configclass
class TraceIntegratedGatedPlayEnvCfg(TraceIntegratedBalancedPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_gated(self)


@configclass
class TraceRecoveryPhysXEnvCfg(TraceRecoveryEnvCfg):
    """Same-stack backend comparison; keep fixed head as a sensed rigid body.

    PhysX does not expose Newton shape sensing. Its fixed head is therefore
    left unmerged (25 native bodies, same rigid transform and source URDF).
    This is a backend representation difference, not a duplicated/approximate
    contact force. Newton model/importer configuration is not changed.
    """
    def __post_init__(self):
        super().__post_init__()
        from isaaclab_physx.physics import PhysxCfg
        self.sim.physics = PhysxCfg()
        self.sim.use_newton_actuators = False
        self.scene.robot.spawn.merge_fixed_joints = False
        from .physx_compat import spawn_reported_urdf, configure_link_sensors
        self.scene.robot.spawn.func = spawn_reported_urdf
        configure_link_sensors(self.scene,K1_REV1_URDF_PATH)


@configclass
class TraceIntegratedGatedLookaheadEnvCfg(TraceIntegratedGatedEnvCfg):
    """Run 21 + BeyondMimic failure look-ahead (a failure also raises the reset bins up to 2 s before it)
    + StableMimic's exact 50:50 uniform/adaptive reset mix."""
    def __post_init__(self):
        super().__post_init__()
        self.commands.reference_trajectory.adaptive_kernel_seconds = 2.
        self.commands.reference_trajectory.adaptive_kernel_decay = .8
        self.commands.reference_trajectory.adaptive_inner_uniform = 0.  # StableMimic 50:50



# Prone get-up without its first 80 frames (0-79: net joint change <= .07 rad, pelvis flat). Real motion
# then starts at ~frame 28, the same as Supine trim100 (joint displacement > .3 rad from the first frame).
PRONE_TRIM80_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../data/motions/K1_rev1/get_up/prone/prone_getup_trim80.npz"))


@configclass
class TraceIntegratedGatedProneTrimEnvCfg(TraceIntegratedGatedLookaheadEnvCfg):
    """Run 22 + Prone reference trimmed by 80 frames (motion onset matched to trimmed Supine)."""
    def __post_init__(self):
        super().__post_init__()
        self.commands.reference_trajectory.prone_trajectory_file = PRONE_TRIM80_FILE


@configclass
class TraceIntegratedGatedProneTrimPlayEnvCfg(TraceIntegratedGatedPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.reference_trajectory.prone_trajectory_file = PRONE_TRIM80_FILE
