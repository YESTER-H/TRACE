# SPDX-License-Identifier: Apache-2.0
"""U1.5 integrated TRACE on the Newton Recovery-only port.

Ports the PhysX U1.4/U1.5 state machine (stable_mimic proposal_transition_commands
and bounded_post_tracking_commands): Tracking/Recovery/Transition regimes chosen
50/50 at reset, oracle w_T, independent Tracking sampler, 1.5 s soft handoff from
Recovery completion or Tracking reacquisition, and a 2 s post-transition Tracking
horizon. Recovery itself keeps the Recovery-only (paper-aligned) MDP unchanged.

Newton differences: Recovery completion is the pre-progression final-frame +
upright test the Recovery-only task terminated on (PhysX switched after
progression), and the paper-aligned success bonus is paid on the first Transition
action, whether the handoff came from completion or from reacquisition.
"""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude
from cyclo_lab.manager_based.mimic.mdp import rewards as mimic_rewards
from .body_adapter import tensor
from .commands import RecoveryOnlyCommand, RecoveryOnlyCommandCfg

TRACKING, RECOVERY, TRANSITION = 0, 1, 2
MOTIONS = ("supine", "prone")


class TraceIntegratedCommand(RecoveryOnlyCommand):
    def __init__(self, cfg, env):
        # Read by the target properties during the base constructor's alignment cache.
        self.tracking_offset_w = torch.zeros((env.num_envs,3), device=env.device)
        super().__init__(cfg, env)
        transition = cfg.transition_s/env.step_dt
        post = cfg.post_tracking_s/env.step_dt
        if (not math.isclose(transition,round(transition),abs_tol=1e-6) or round(transition)<2 or
                not math.isclose(post,round(post),abs_tol=1e-6) or round(post)<1):
            raise ValueError("Transition (>=2) and post-Tracking (>=1) must span integer control steps")
        if cfg.reacquisition_persistence_steps < 1:
            raise ValueError("Reacquisition persistence must be positive")
        self.transition_steps_required = round(transition)
        self.post_tracking_steps_required = round(post)
        self.episode_origins = self.regimes.clone()
        self.transition_steps = torch.zeros_like(self.regimes)
        self.reacquisition_steps = torch.zeros_like(self.regimes)
        self.reacquisition_candidate = torch.zeros_like(self.just_reset)
        self.post_tracking_steps = torch.zeros_like(self.regimes)
        self.post_tracking_active = torch.zeros_like(self.just_reset)
        self.post_tracking_completion = torch.zeros_like(self.just_reset)
        self.upright_steps = torch.zeros_like(self.regimes)
        self.recovered_like = torch.zeros_like(self.just_reset)
        self._gate_prev_regimes = self.regimes.clone()
        self._gate_target = torch.zeros((self.num_envs,3), device=self.device)
        self._last_gate_step = -1
        self.forced_regimes = torch.full_like(self.regimes, -1)
        self.tracking_failure_scores = torch.zeros(20, device=self.device)
        self.tracking_reset_bins = torch.zeros(20, dtype=torch.long, device=self.device)
        self.tracking_failure_bins = torch.zeros_like(self.tracking_reset_bins)
        self.end_body_ids = [cfg.body_names.index(n) for n in cfg.end_body_names]
        names = ["tracking_episode_resets", "tracking_uniform_resets", "tracking_adaptive_resets",
                 "tracking_failures", "tracking_origin_failures", "post_tracking_failures",
                 "transition_complete", "post_tracking_enters"]
        for m in MOTIONS:
            names += [f"{m}_transition_from_terminal", f"{m}_transition_from_reacquisition",
                      f"{m}_transition_from_early_upright",
                      f"{m}_transition_completes", f"{m}_post_tracking_completions",
                      f"{m}_post_tracking_interruptions"]
        for name in names:
            self.events[name] = torch.zeros((), dtype=torch.long, device=self.device)
            self.pending_events[name] = torch.zeros_like(self.events[name])
        self._last_horizon_step = -1

    # ------------------------------------------------------------------
    # Targets: live Tracking frame (Tracking/Transition) or hidden Recovery k+1.
    # ------------------------------------------------------------------
    def _raw(self, name, successor=True):
        out = super()._raw(name, successor)
        tracking = self.regimes != RECOVERY
        out[tracking] = getattr(self.reference,name)[self.tracking_phase[tracking]]
        return out

    @property
    def body_pos_w(self):
        offset = torch.where((self.regimes!=RECOVERY)[:,None], self.tracking_offset_w, 0.)
        return super().body_pos_w + offset[:,None]

    @property
    def live_body_pos_w(self):
        origin = tensor(self._env.scene.env_origins)+self.tracking_offset_w
        return self.reference.body_position_w[self.tracking_phase][:,self.tracked_ids] + origin[:,None]

    @property
    def live_body_quat_w(self):
        return self.reference.body_orientation_w[self.tracking_phase][:,self.tracked_ids]

    @property
    def live_anchor_pos_w(self):
        return self.live_body_pos_w[:,self.reference_anchor_body_id]

    @property
    def live_anchor_quat_w(self):
        return self.live_body_quat_w[:,self.reference_anchor_body_id]

    @property
    def transition_alpha(self):
        # Exactly transition_steps_required executed Transition actions carry alpha 0 ... 1.
        return (self.transition_steps.float()/(self.transition_steps_required-1)).clamp(0,1)

    @property
    def tracking_weight(self):
        return torch.where(self.regimes==TRACKING, 1.,
                           torch.where(self.regimes==TRANSITION, self.transition_alpha, 0.))

    @property
    def imitation_scale(self):
        return torch.where(self.regimes==RECOVERY, self.cfg.recovery_imitation_scale,
                           torch.where(self.regimes==TRANSITION, self.transition_alpha, 1.))

    def final_mask(self):
        return super().final_mask()&(self.regimes==RECOVERY)

    def recovery_target_error(self):
        return torch.where(self.regimes==RECOVERY, super().recovery_target_error(), 0.)

    # ------------------------------------------------------------------
    # Reset: 50/50 regime, unchanged Recovery sampler, independent Tracking sampler.
    # ------------------------------------------------------------------
    def tracking_bins(self, phases):
        return (phases*20//self.reference.num_frames).clamp_max(19)

    def tracking_adaptive_probabilities(self):
        prioritized = torch.softmax(torch.log(self.tracking_failure_scores+1e-3),0)
        return .8*prioritized+.2/20

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if not len(ids):
            return
        p = torch.tensor(self.cfg.reset_regime_probabilities,device=self.device)
        regimes = torch.multinomial(p/p.sum(),len(ids),replacement=True)
        forced = self.forced_regimes[ids]
        regimes[forced>=0] = forced[forced>=0]
        if torch.any((regimes!=TRACKING)&(regimes!=RECOVERY)):
            raise ValueError("Direct resets start in Tracking or Recovery only")
        self.regimes[ids] = regimes
        self.episode_origins[ids] = regimes
        self.tracking_offset_w[ids] = 0.
        recovery_ids = ids[regimes==RECOVERY]
        super()._resample_command(recovery_ids)
        # StableMimic gate safeguard: a reset already near the command height keeps the
        # Tracking gate target for its Recovery segment (reference heights, before noise).
        self.recovered_like[ids] = False
        if len(recovery_ids):
            height = torch.zeros(len(recovery_ids), device=self.device)
            identities = self.recovery_reference_ids[recovery_ids]
            for i,ref in enumerate(self.recovery_references):
                mine = identities==i
                height[mine] = ref.body_position_w[self.recovery_phase[recovery_ids[mine]]][:,0,2]
            command_height = self.reference.body_position_w[self.tracking_phase[recovery_ids]][:,0,2]
            self.recovered_like[recovery_ids] = height>=self.cfg.recovered_like_height_ratio*command_height
        tracking_ids = ids[regimes==TRACKING]
        if len(tracking_ids):
            n = len(tracking_ids)
            frames = torch.randint(self.reference.num_frames,(n,),device=self.device)
            adaptive = torch.rand(n,device=self.device)>=self.cfg.tracking_uniform_sampling_probability
            count = int(adaptive.sum())
            bins = (torch.multinomial(self.tracking_adaptive_probabilities(),count,replacement=True) if count
                    else torch.empty(0,dtype=torch.long,device=self.device))
            # Ceil boundaries invert floor(frame*bins/frames), as in U1.4.
            lo = (bins*self.reference.num_frames+19)//20
            hi = ((bins+1)*self.reference.num_frames+19)//20
            frames[adaptive] = lo+(torch.rand(len(bins),device=self.device)*(hi-lo)).long()
            if self.cfg.tracking_start_from_zero:
                frames.zero_()
            self.tracking_phase[tracking_ids] = frames
            self._event("tracking_episode_resets",n)
            self._event("tracking_adaptive_resets",adaptive.sum())
            self._event("tracking_uniform_resets",n-adaptive.sum())
            self.tracking_reset_bins += torch.bincount(self.tracking_bins(frames),minlength=20)
            self.write_reference_state(tracking_ids,self.reference,frames,perturb=self.cfg.reset_perturbations,
                                       pose_noise=self.cfg.tracking_reset_pose_noise,
                                       velocity_noise=self.cfg.tracking_reset_velocity_noise,
                                       joint_position_noise=self.cfg.tracking_reset_joint_position_noise,
                                       joint_velocity_noise=0.,floor_clamp=False)
            self.last_direct_reset_branch[tracking_ids] = -1
            self.just_reset[tracking_ids] = True
            self.hold_steps[tracking_ids] = 0
            self._clear_persistent(tracking_ids)
            self.forced_reference_ids[tracking_ids] = -1
            self.forced_frames[tracking_ids] = -1
        self.transition_steps[ids] = 0
        self.reacquisition_steps[ids] = 0
        self.reacquisition_candidate[ids] = False
        self.post_tracking_steps[ids] = 0
        self.post_tracking_active[ids] = False
        self.post_tracking_completion[ids] = False
        self.upright_steps[ids] = 0
        self._last_gate_step = -1  # an env.reset() outside step() must relabel
        self.forced_regimes[ids] = -1
        self._update_alignment_cache()

    def persistent_exempt_mask(self):
        exempt = super().persistent_exempt_mask()
        if self.cfg.early_upright_steps > 0:  # standing ahead of the reference is not a failure
            exempt |= self.upright_mask()&(self.regimes==RECOVERY)
        return exempt

    def _clear_persistent(self, ids):
        self.persistent_steps[ids] = 0
        self.failure_onset_phase[ids] = -1
        self.failure_detection_phase[ids] = -1
        self._persistent_result[ids] = False

    # ------------------------------------------------------------------
    # Handoff: Recovery completion / reacquisition -> Transition -> Tracking.
    # ------------------------------------------------------------------
    def _align_live_xy(self, ids):
        self.tracking_offset_w[ids,:2] += self.robot_anchor_pos_w[ids,:2]-self.live_anchor_pos_w[ids,:2]

    def _enter_transition(self, ids, reason):
        if not len(ids):
            return
        self.regimes[ids] = TRANSITION
        self.transition_steps[ids] = 0
        self.upright_steps[ids] = 0
        self.hold_steps[ids] = 0
        self.reacquisition_steps[ids] = 0
        self._clear_persistent(ids)
        self._align_live_xy(ids)
        for i,m in enumerate(MOTIONS):
            self._event(f"{m}_transition_from_{reason}",(self.recovery_reference_ids[ids]==i).sum())

    def reacquisition_errors(self):
        reference = self.live_body_pos_w.clone()
        delta = self.robot_anchor_pos_w-self.live_anchor_pos_w
        delta[:,2] = 0
        reference += delta[:,None]
        joint_pos = tensor(self.robot.data.joint_pos)[:,self.joint_ids]
        joint_vel = tensor(self.robot.data.joint_vel)[:,self.joint_ids]
        live_lin = self.reference.body_linear_velocity_w[self.tracking_phase][:,self.tracked_ids]
        live_ang = self.reference.body_angular_velocity_w[self.tracking_phase][:,self.tracked_ids]
        rms = lambda x: x.square().mean(-1).sqrt()
        return {
            "joint_position": rms(self.reference.joint_position[self.tracking_phase]-joint_pos),
            "joint_velocity": rms(self.reference.joint_velocity[self.tracking_phase]-joint_vel),
            "anchor_orientation": quat_error_magnitude(self.live_anchor_quat_w,self.robot_anchor_quat_w),
            "root_height": (reference[:,0,2]-self.robot_body_pos_w[:,0,2]).abs(),
            "body_position": rms((reference-self.robot_body_pos_w).norm(dim=-1)),
            "body_orientation": rms(quat_error_magnitude(self.live_body_quat_w,self.robot_body_quat_w)),
            "body_linear_velocity": rms((live_lin-self.robot_body_lin_vel_w).norm(dim=-1)),
            "body_angular_velocity": rms((live_ang-self.robot_body_ang_vel_w).norm(dim=-1)),
        }

    def reacquisition_mask(self, errors):
        # Every component must pass; a good joint pose cannot hide a bad tilt.
        return torch.stack([errors[k]<=v for k,v in self.cfg.reacquisition_scales.items()]).all(0)

    def _update_command(self):
        active = ~self.just_reset
        old_transition = (self.regimes==TRANSITION)&active
        completed = self.success_mask()&active
        self.tracking_phase[active] = (self.tracking_phase[active]+1)%self.reference.num_frames
        recovery = (self.regimes==RECOVERY)&active&~completed
        was_final = self.final_mask()
        for i,ref in enumerate(self.recovery_references):
            ids = recovery&(self.recovery_reference_ids==i)
            self.recovery_phase[ids] = (self.recovery_phase[ids]+1).clamp_max(ref.num_frames-1)
        self.hold_steps[was_final&active] += 1
        self.hold_steps[~was_final] = 0
        self._enter_transition(torch.where(completed)[0],"terminal")
        upright = self.upright_mask()&(self.regimes==RECOVERY)&active
        self.upright_steps[:] = torch.where(upright,self.upright_steps+1,0)
        if self.cfg.early_upright_steps > 0:
            early = upright&(self.upright_steps>=self.cfg.early_upright_steps)
            self._enter_transition(torch.where(early)[0],"early_upright")
        self.transition_steps[old_transition] += 1
        done = torch.where(old_transition&(self.transition_steps>=self.transition_steps_required))[0]
        if len(done):
            self.regimes[done] = TRACKING
            self._align_live_xy(done)
            # Transition only starts from Recovery, so every completion opens the U1.5 horizon.
            self.post_tracking_active[done] = True
            self.post_tracking_steps[done] = 0
            self._event("transition_complete",len(done))
            self._event("post_tracking_enters",len(done))
            for i,m in enumerate(MOTIONS):
                self._event(f"{m}_transition_completes",(self.recovery_reference_ids[done]==i).sum())
        candidate = self.reacquisition_mask(self.reacquisition_errors())&(self.regimes==RECOVERY)&active
        self.reacquisition_candidate[:] = candidate
        self.reacquisition_steps[:] = torch.where(candidate,self.reacquisition_steps+1,0)
        confirmed = candidate&(self.reacquisition_steps>=self.cfg.reacquisition_persistence_steps)
        self._enter_transition(torch.where(confirmed)[0],"reacquisition")
        self.just_reset.zero_()
        self._update_alignment_cache()

    def gate_supervision(self):
        """Training-only gate labels [y_T, same regime as previous step, Transition].

        y_T is the oracle Tracking weight (1 / 0 / alpha); a recovered-like reset keeps 1.
        Computed once per control step (the observation after a reset sees just_reset).
        """
        step = self._env.common_step_counter
        if step != self._last_gate_step:
            self._last_gate_step = step
            target = torch.where((self.regimes==RECOVERY)&self.recovered_like, 1., self.tracking_weight)
            same = (self.regimes==self._gate_prev_regimes)&~self.just_reset
            self._gate_target = torch.stack((target, same.float(), (self.regimes==TRANSITION).float()), -1)
            self._gate_prev_regimes = self.regimes.clone()
        return self._gate_target

    # ------------------------------------------------------------------
    # Tracking failure and the bounded post-transition horizon (terminations).
    # ------------------------------------------------------------------
    def tracking_failure(self):
        gravity = self.robot_anchor_quat_w.new_tensor((0.,0.,-1.)).expand(self.num_envs,3)
        pelvis_height = self.logical_body_pos_w[:,0,2]-tensor(self._env.scene.env_origins)[:,2]
        alignment = -tensor(self.robot.data.projected_gravity_b)[:,2]
        fallen = (pelvis_height<=self.cfg.fall_pelvis_height)&(alignment<=self.cfg.fall_gravity_alignment)
        ref_g = quat_apply_inverse(self.live_anchor_quat_w,gravity)
        actual_g = quat_apply_inverse(self.robot_anchor_quat_w,gravity)
        deviated = (self.live_anchor_pos_w[:,2]-self.robot_anchor_pos_w[:,2]).abs()>self.cfg.tracking_anchor_height_threshold
        deviated |= (ref_g[:,2]-actual_g[:,2]).abs()>self.cfg.tracking_gravity_threshold
        end = self.live_body_pos_w[:,self.end_body_ids,2]-self.robot_body_pos_w[:,self.end_body_ids,2]
        deviated |= end.abs().amax(-1)>self.cfg.tracking_end_body_height_threshold
        failed = (fallen|deviated)&(self.regimes==TRACKING)
        ids = torch.where(failed)[0]
        if len(ids):
            bins = self.tracking_bins(self.tracking_phase[ids])
            hist = torch.bincount(bins,minlength=20)
            self.tracking_failure_bins += hist
            self.tracking_failure_scores.mul_(.9).add_(.1*hist.float()/hist.sum())
            origin = self.episode_origins[ids]
            self._event("tracking_failures",len(ids))
            self._event("tracking_origin_failures",(origin==TRACKING).sum())
            self._event("post_tracking_failures",(origin==RECOVERY).sum())
        return failed

    def update_post_tracking_horizon(self):
        """Run LAST among terminations: earlier failures/time-outs win, as in U1.5."""
        step = self._env.common_step_counter
        if step == self._last_horizon_step:
            return self.post_tracking_completion
        self._last_horizon_step = step
        post = self.post_tracking_active&(self.regimes==TRACKING)
        self.post_tracking_steps[post] += 1
        manager = self._env.termination_manager
        interrupted = post&(manager.terminated|manager.time_outs)
        complete = post&~interrupted&(self.post_tracking_steps>=self.post_tracking_steps_required)
        self.post_tracking_completion[:] = complete
        for i,m in enumerate(MOTIONS):
            mine = self.recovery_reference_ids==i
            self._event(f"{m}_post_tracking_completions",(complete&mine).sum())
            self._event(f"{m}_post_tracking_interruptions",(interrupted&mine).sum())
        return complete


@configclass
class TraceIntegratedCommandCfg(RecoveryOnlyCommandCfg):
    class_type: type = TraceIntegratedCommand
    reset_regime_probabilities: tuple[float,float] = (.5,.5)  # (Tracking, Recovery)
    recovery_imitation_scale: float = 2.5
    # Tracking direct resets: U1.4 sampler and the original Mimic Tracking reset ranges.
    tracking_uniform_sampling_probability: float = .5
    tracking_start_from_zero: bool = False
    tracking_reset_pose_noise: dict = {"x":(-.05,.05),"y":(-.05,.05),"z":(-.01,.01),
                                       "roll":(-.1,.1),"pitch":(-.1,.1),"yaw":(-.2,.2)}
    tracking_reset_velocity_noise: dict = {"x":(-.5,.5),"y":(-.5,.5),"z":(-.2,.2),
                                           "roll":(-.52,.52),"pitch":(-.52,.52),"yaw":(-.78,.78)}
    tracking_reset_joint_position_noise: float = .1
    # Tracking failure: U1.4 deviation limits plus the U1 fall detector.
    end_body_names: list[str] = ["left_ankle_roll_link","right_ankle_roll_link",
                                 "left_wrist_roll_rubber_hand","right_wrist_roll_rubber_hand"]
    tracking_anchor_height_threshold: float = .25
    tracking_gravity_threshold: float = .8
    tracking_end_body_height_threshold: float = .25
    fall_pelvis_height: float = .514
    fall_gravity_alignment: float = .2
    transition_s: float = 1.5
    post_tracking_s: float = 2.
    reacquisition_persistence_steps: int = 5
    # Recovery -> Transition once upright (pelvis/tilt test) this many consecutive steps,
    # before the final frame; upright Recovery states are then exempt from the 2 s error rule.
    early_upright_steps: int = 0  # 0: off (runs 18-20)
    recovered_like_height_ratio: float = .8
    reacquisition_scales: dict = {
        "joint_position":1., "joint_velocity":4., "root_height":.25, "anchor_orientation":.4,
        "body_position":.3, "body_orientation":.4, "body_linear_velocity":1., "body_angular_velocity":3.14,
    }


def command(env, name="reference_trajectory"):
    return env.command_manager.get_term(name)


def oracle_tracking_weight(env):
    return command(env).tracking_weight[:,None]


def integrated_imitation_reward(env, command_name, std, kernel_name):
    """Tracking kernels x1, Recovery x2.5 (root height for the anchor term), Transition x alpha."""
    c = command(env,command_name)
    value = getattr(mimic_rewards,kernel_name)(env,command_name,std)
    if kernel_name == "reference_anchor_position_tracking":
        height = torch.exp(-(c.body_pos_w[:,0,2]-c.robot_body_pos_w[:,0,2]).square()/std**2)
        value = torch.where(c.regimes==RECOVERY,height,value)
    return value*c.imitation_scale


def gate_supervision(env):
    return command(env).gate_supervision()


def handoff_bonus(env):
    """Recovery success bonus on the first Transition action, for completion and reacquisition alike.

    Paying only the terminal completion would make an earlier reacquisition forfeit it.
    """
    c = command(env)
    return ((c.regimes==TRANSITION)&(c.transition_steps==0)).float()


def tracking_failure(env):
    return command(env).tracking_failure()


def post_tracking_horizon_complete(env):
    return command(env).update_post_tracking_horizon()


def tracking_origin_horizon_complete(env, horizon_s):
    """Time-out for Tracking-origin episodes only; Recovery-origin episodes keep the global limit."""
    steps = round(horizon_s/env.step_dt)
    return (command(env).episode_origins==TRACKING)&(env.episode_length_buf>=steps)
