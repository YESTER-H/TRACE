# SPDX-License-Identifier: Apache-2.0
"""Recovery-only semantic port: hidden successor targets and U1.2 onset sampler."""
import math
from dataclasses import MISSING
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_inv, quat_apply, quat_error_magnitude, yaw_quat, quat_from_euler_xyz
from cyclo_lab.manager_based.mimic.mdp.reference_trajectory import ReferenceTrajectory
from .body_adapter import BODY_NAMES, JOINT_NAMES, LogicalBodyAdapter, tensor, refresh_reset_cache, rotate


class RecoveryOnlyCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.adapter = LogicalBodyAdapter(self.robot, cfg.urdf_path)
        if set(self.robot.joint_names) != set(JOINT_NAMES):
            raise ValueError("K1 must have exactly the canonical 23 named joints")
        self.joint_ids = torch.tensor([self.robot.joint_names.index(n) for n in JOINT_NAMES], device=self.device)
        self.tracked_ids = torch.tensor([BODY_NAMES.index(n) for n in cfg.body_names], device=self.device)
        self.reference_anchor_body_id = cfg.body_names.index(cfg.anchor_body_name)
        self.logical_anchor_id = BODY_NAMES.index(cfg.anchor_body_name)
        def load(path):
            return ReferenceTrajectory(path, runtime_joint_names=JOINT_NAMES, tracked_body_names=BODY_NAMES,
                                       source_joint_names=JOINT_NAMES, source_body_names=BODY_NAMES,
                                       source_quat_order="wxyz", expected_fps=1/env.step_dt, device=self.device)
        self.reference = load(cfg.trajectory_file)
        self.recovery_references = [load(cfg.supine_trajectory_file), load(cfg.prone_trajectory_file)]
        self.reference_names = ("Supine", "Prone")
        self.tracking_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.recovery_reference_ids = torch.zeros_like(self.tracking_phase)
        self.recovery_phase = torch.zeros_like(self.tracking_phase)
        self.regimes = torch.ones_like(self.tracking_phase)  # Recovery-only, not an integrated state machine.
        self.just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_steps = torch.zeros_like(self.tracking_phase)
        self.persistent_steps = torch.zeros_like(self.tracking_phase)
        self.failure_onset_phase = torch.full_like(self.tracking_phase, -1)
        self.failure_detection_phase = torch.full_like(self.tracking_phase, -1)
        self.failure_score_by_reference_bin = torch.zeros((2,20), device=self.device)
        self.last_direct_reset_branch = torch.full_like(self.tracking_phase, -1)
        self.forced_reference_ids = torch.full_like(self.tracking_phase, -1)
        self.forced_frames = torch.full_like(self.tracking_phase, -1)
        self.events = {name: torch.zeros((), dtype=torch.long, device=self.device) for name in
                       ("supine_episode_resets","prone_episode_resets","uniform_episode_resets",
                        "adaptive_episode_resets","success_completions","persistent_failures","grace_failures")}
        self.pending_events = {name: torch.zeros_like(value) for name,value in self.events.items()}
        self.reset_bins = torch.zeros((2,20), dtype=torch.long, device=self.device)
        self.failure_onset_bins = torch.zeros_like(self.reset_bins)
        self.failure_detection_bins = torch.zeros_like(self.reset_bins)
        self.persistent_error_steps_required = round(cfg.persistence_s/env.step_dt)
        self.grace_steps = round(cfg.grace_s/env.step_dt)
        if (cfg.phase_bins != 20 or not math.isclose(cfg.persistence_s/env.step_dt,100) or
                not math.isclose(cfg.grace_s/env.step_dt,self.grace_steps) or self.grace_steps<1):
            raise ValueError("Require 20 bins, integer grace and 100 control-step persistence")
        self._last_persistent_step = -1
        self._persistent_result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.adaptive_kernels = [self._lookahead_kernel(ref.num_frames,env.step_dt) for ref in self.recovery_references]
        self._states = None
        self._states_time = None
        self._update_alignment_cache()

    @property
    def recovery_reward_phase(self):
        out = self.recovery_phase.clone()
        for i,ref in enumerate(self.recovery_references):
            mask = self.recovery_reference_ids == i
            out[mask] = (out[mask]+1).clamp_max(ref.num_frames-1)
        return out

    @property
    def command(self):
        return torch.cat((self.joint_pos,self.joint_vel),-1)

    def _raw(self, name, successor=True):
        phases = self.recovery_reward_phase if successor else self.recovery_phase
        ref = self.recovery_references[0]
        shape = getattr(ref,name).shape[1:]
        out = torch.empty((self.num_envs,*shape),device=self.device)
        for i,reference in enumerate(self.recovery_references):
            mask = self.recovery_reference_ids == i
            out[mask] = getattr(reference,name)[phases[mask]]
        return out

    @property
    def joint_pos(self):
        return self._raw("joint_position")

    @property
    def joint_vel(self):
        return self._raw("joint_velocity")

    @property
    def body_pos_w(self):
        return self._raw("body_position_w")[:,self.tracked_ids] + tensor(self._env.scene.env_origins)[:,None]

    @property
    def body_quat_w(self):
        return self._raw("body_orientation_w")[:,self.tracked_ids]

    @property
    def body_lin_vel_w(self):
        return self._raw("body_linear_velocity_w")[:,self.tracked_ids]

    @property
    def body_ang_vel_w(self):
        return self._raw("body_angular_velocity_w")[:,self.tracked_ids]

    @property
    def anchor_pos_w(self):
        return self.body_pos_w[:,self.reference_anchor_body_id]

    @property
    def anchor_quat_w(self):
        return self.body_quat_w[:,self.reference_anchor_body_id]

    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w[:,self.reference_anchor_body_id]

    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w[:,self.reference_anchor_body_id]

    def logical_states(self):
        timestamp = self.robot.data._sim_timestamp
        if self._states is None or timestamp != self._states_time:
            self._states = self.adapter.states()
            self._states_time = timestamp
        return self._states

    @property
    def logical_body_pos_w(self):
        return self.logical_states()[0]

    @property
    def logical_body_quat_w(self):
        return self.logical_states()[1]

    @property
    def logical_body_lin_vel_w(self):
        return self.logical_states()[2]

    @property
    def logical_body_ang_vel_w(self):
        return self.logical_states()[3]

    @property
    def robot_body_pos_w(self):
        return self.logical_body_pos_w[:,self.tracked_ids]

    @property
    def robot_body_quat_w(self):
        return self.logical_body_quat_w[:,self.tracked_ids]

    @property
    def robot_body_lin_vel_w(self):
        return self.logical_body_lin_vel_w[:,self.tracked_ids]

    @property
    def robot_body_ang_vel_w(self):
        return self.logical_body_ang_vel_w[:,self.tracked_ids]

    @property
    def robot_anchor_pos_w(self):
        return self.logical_body_pos_w[:,self.logical_anchor_id]

    @property
    def robot_anchor_quat_w(self):
        return self.logical_body_quat_w[:,self.logical_anchor_id]

    @property
    def robot_anchor_lin_vel_w(self):
        return self.logical_body_lin_vel_w[:,self.logical_anchor_id]

    @property
    def robot_anchor_ang_vel_w(self):
        return self.logical_body_ang_vel_w[:,self.logical_anchor_id]

    def phase_bins(self, identities=None, phases=None):
        identities = self.recovery_reference_ids if identities is None else identities
        phases = self.recovery_phase if phases is None else phases
        out = torch.empty_like(phases)
        for i,ref in enumerate(self.recovery_references):
            mask = identities == i
            out[mask] = (phases[mask]*20//(ref.num_frames-1)).clamp_max(19)
        return out

    def _lookahead_kernel(self, num_frames, dt):
        """BeyondMimic's non-causal smoothing in seconds: bin s also takes failures recorded up to
        adaptive_kernel_seconds later, weighted decay**(seconds ahead). [20, 20] matrix, or None (off).

        BeyondMimic (1 s bins, u in {0,1,2}, rho .8): p_s = sum_u rho**u f_{s+u}, the last bin
        replicated past the end. Our bins are (num_frames-1)/20 frames long, so u runs in bins while
        the window and the decay stay in seconds.
        """
        if self.cfg.adaptive_kernel_seconds <= 0:
            return None
        bin_s = (num_frames-1)/20*dt
        kernel = torch.zeros((20,20), device=self.device)
        u = 0
        while u*bin_s <= self.cfg.adaptive_kernel_seconds+1e-9:
            for s in range(20):
                kernel[s,min(s+u,19)] += self.cfg.adaptive_kernel_decay**(u*bin_s)
            u += 1
        return kernel

    def adaptive_bin_probabilities(self, identity):
        score = self.failure_score_by_reference_bin[identity]
        if self.adaptive_kernels[identity] is not None:
            score = self.adaptive_kernels[identity]@score
            score = score/score.sum().clamp_min(1e-12)
        prioritized = torch.softmax(torch.log(score+1e-3),0)
        mix = self.cfg.adaptive_inner_uniform
        return (1-mix)*prioritized+mix*torch.full_like(prioritized,1/20)

    def _event(self,name,count):
        self.events[name] += count
        self.pending_events[name] += count

    def write_reference_state(self, ids, ref, frames, perturb=False, pose_noise=None, velocity_noise=None,
                              joint_position_noise=None, joint_velocity_noise=None, floor_clamp=True):
        """Root link pose + actual COM velocity; no physics step is added.

        Noise arguments default to the Recovery cfg; integrated Tracking resets pass their own.
        """
        p = ref.body_position_w[frames,0].clone()+tensor(self._env.scene.env_origins)[ids]
        q = ref.body_orientation_w[frames,0].clone()
        v = ref.body_linear_velocity_w[frames,0].clone()
        w = ref.body_angular_velocity_w[frames,0].clone()
        jq,jv = ref.joint_position[frames].clone(),ref.joint_velocity[frames].clone()
        if perturb:
            def ranges(mapping):
                bounds = p.new_tensor([mapping.get(k,(0.,0.)) for k in ("x","y","z","roll","pitch","yaw")])
                return bounds[:,0]+torch.rand((len(ids),6),device=self.device)*(bounds[:,1]-bounds[:,0])
            noise = ranges(self.cfg.reset_pose_noise if pose_noise is None else pose_noise)
            if floor_clamp:
                minimum = ref.body_position_w[frames][:,self.tracked_ids,2].amin(-1)
                noise[:,2] = torch.maximum(noise[:,2],self.cfg.floor_clearance-minimum)
            p += noise[:,:3]
            q = quat_mul(quat_from_euler_xyz(*noise[:,3:].unbind(-1)),q)
            q = q/q.norm(dim=-1,keepdim=True)
            noise_v = ranges(self.cfg.reset_velocity_noise if velocity_noise is None else velocity_noise)
            v += noise_v[:,:3]
            w += noise_v[:,3:]
            if joint_position_noise is None:
                joint_position_noise = self.cfg.reset_joint_position_noise
            if joint_velocity_noise is None:
                joint_velocity_noise = self.cfg.reset_joint_velocity_noise
            jq += (torch.rand_like(jq)*2-1)*joint_position_noise
            jv += (torch.rand_like(jv)*2-1)*joint_velocity_noise
            limits = tensor(self.robot.data.soft_joint_pos_limits)[ids][:,self.joint_ids]
            jq = jq.clamp(limits[...,0],limits[...,1])
        pelvis = self.robot.body_names.index("pelvis")
        actual_com = tensor(self.robot.data.body_com_pos_b)[ids,pelvis]
        v += torch.cross(w,rotate(q,actual_com-self.adapter.source_com[0]),dim=-1)
        # The shared Warp state-write contract is int32; the PhysX backend
        # does not coerce torch int64 indices (Newton happens to do so).
        write_ids = ids.to(torch.int32)
        write_joints = self.joint_ids.to(torch.int32)
        self.robot.write_joint_state_to_sim_index(position=jq.contiguous(),velocity=jv.contiguous(),
                                                  env_ids=write_ids,joint_ids=write_joints)
        self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((p,q),-1).contiguous(),env_ids=write_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.cat((v,w),-1).contiguous(),env_ids=write_ids)
        refresh_reset_cache(self.robot)
        self._states = None
        return p,q,jq,jv

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if not len(ids):
            return
        identities = torch.randint(2,(len(ids),),device=self.device)
        forced = self.forced_reference_ids[ids]
        identities[forced>=0] = forced[forced>=0]
        if torch.any((identities<0)|(identities>1)):
            raise ValueError("Invalid forced reference identity")
        self.recovery_reference_ids[ids] = identities
        self.tracking_phase[ids] = torch.randint(self.reference.num_frames,(len(ids),),device=self.device)
        self.last_direct_reset_branch[ids] = -1
        for identity,ref in enumerate(self.recovery_references):
            selected = ids[identities==identity]
            if not len(selected):
                continue
            frames = torch.empty(len(selected),device=self.device,dtype=torch.long)
            force = self.forced_frames[selected]
            forced_mask = force>=0
            frames[forced_mask] = force[forced_mask]
            random_ids = torch.where(~forced_mask)[0]
            uniform = torch.rand(len(random_ids),device=self.device)<.5
            ui,ai = random_ids[uniform],random_ids[~uniform]
            frames[ui] = torch.randint(ref.num_frames-1,(len(ui),),device=self.device)
            if len(ai):
                bins = torch.multinomial(self.adaptive_bin_probabilities(identity),len(ai),replacement=True)
                lower,upper = bins*(ref.num_frames-1)//20,(bins+1)*(ref.num_frames-1)//20
                frames[ai] = lower+torch.floor(torch.rand(len(ai),device=self.device)*(upper-lower)).long()
            if torch.any((frames<0)|(frames>=ref.num_frames-1)):
                raise ValueError("Direct resets must have a valid k+1 successor")
            self.last_direct_reset_branch[selected[ui]] = 0
            self.last_direct_reset_branch[selected[ai]] = 1
            self._event("uniform_episode_resets",len(ui))
            self._event("adaptive_episode_resets",len(ai))
            self._event(("supine_episode_resets","prone_episode_resets")[identity],len(selected))
            self.recovery_phase[selected] = frames
            self.reset_bins[identity] += torch.bincount(self.phase_bins(torch.full_like(frames,identity),frames),minlength=20)
            self.write_reference_state(selected,ref,frames,perturb=self.cfg.reset_perturbations)
        self.forced_reference_ids[ids] = -1
        self.forced_frames[ids] = -1
        self.just_reset[ids] = True
        self.hold_steps[ids] = 0
        self.persistent_steps[ids] = 0
        self.failure_onset_phase[ids] = -1
        self.failure_detection_phase[ids] = -1
        self._persistent_result[ids] = False
        self._update_alignment_cache()

    def final_mask(self):
        result = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        for i,ref in enumerate(self.recovery_references):
            result |= (self.recovery_reference_ids==i)&(self.recovery_phase==ref.num_frames-1)
        return result

    def upright_mask(self):
        gravity = tensor(self.robot.data.projected_gravity_b)
        return (self.logical_body_pos_w[:,0,2]>=self.cfg.upright_height)&(-gravity[:,2]>=self.cfg.upright_alignment)

    def success_mask(self):
        finite = torch.isfinite(tensor(self.robot.data.joint_pos)).all(-1)&torch.isfinite(tensor(self.robot.data.joint_vel)).all(-1)
        safe = tensor(self.robot.data.joint_vel).abs().amax(-1)<=100
        return self.final_mask()&self.upright_mask()&finite&safe

    def persistent_exempt_mask(self):
        """States the 2 s error rule never fails (here: completion only)."""
        return self.success_mask()

    def target_error_components(self):
        return {
            "root_height":(self.body_pos_w[:,0,2]-self.robot_body_pos_w[:,0,2]).abs(),
            "anchor_orientation":quat_error_magnitude(self.anchor_quat_w,self.robot_anchor_quat_w),
            "body_position":(self.aligned_body_position_w-self.robot_body_pos_w).norm(dim=-1).mean(-1),
            "body_orientation":quat_error_magnitude(self.aligned_body_orientation_w,self.robot_body_quat_w).mean(-1),
        }

    def recovery_target_error(self):
        c = self.target_error_components()
        return torch.stack(((c["root_height"]/.3).square(),(c["anchor_orientation"]/.4).square(),
                            (c["body_position"]/.3).square(),(c["body_orientation"]/.4).square()),-1).mean(-1)

    def update_persistent_target_error(self):
        # Terminations are evaluated once before command progression each policy step.
        step = self._env.common_step_counter
        if step == self._last_persistent_step:
            return self._persistent_result
        self._last_persistent_step = step
        violation = self.recovery_target_error()>4
        starts = violation&(self.persistent_steps==0)
        self.failure_onset_phase[starts] = self.recovery_phase[starts]
        self.persistent_steps[violation] += 1
        self.persistent_steps[~violation] = 0
        self.failure_onset_phase[~violation] = -1
        self.failure_detection_phase[~violation] = -1
        failed = violation&(self.persistent_steps>=100)&~self.persistent_exempt_mask()
        new = failed&(self.failure_detection_phase<0)
        ids = torch.where(new)[0]
        self.failure_detection_phase[ids] = self.recovery_phase[ids]
        self._event("persistent_failures",len(ids))
        for i in range(2):
            selected = ids[self.recovery_reference_ids[ids]==i]
            if len(selected):
                onset = self.phase_bins(torch.full_like(selected,i),self.failure_onset_phase[selected])
                detection = self.phase_bins(torch.full_like(selected,i),self.recovery_phase[selected])
                hist = torch.bincount(onset,minlength=20).float()
                self.failure_onset_bins[i] += hist.long()
                self.failure_detection_bins[i] += torch.bincount(detection,minlength=20)
                self.failure_score_by_reference_bin[i] = .9*self.failure_score_by_reference_bin[i]+.1*hist/hist.sum()
        self._persistent_result = failed
        return failed

    def _update_alignment_cache(self):
        ap,aq = self.anchor_pos_w[:,None],self.anchor_quat_w[:,None]
        rp,rq = self.robot_anchor_pos_w[:,None],self.robot_anchor_quat_w[:,None]
        delta = yaw_quat(quat_mul(rq,quat_inv(aq)))
        offset = rp.clone()
        offset[...,2] = ap[...,2]
        self.aligned_body_position_w = offset+quat_apply(delta.expand(-1,len(self.cfg.body_names),-1),self.body_pos_w-ap)
        self.aligned_body_orientation_w = quat_mul(delta.expand(-1,len(self.cfg.body_names),-1),self.body_quat_w)

    def _update_command(self):
        active = ~self.just_reset
        self.tracking_phase[active] = (self.tracking_phase[active]+1)%self.reference.num_frames
        was_final = self.final_mask()
        for i,ref in enumerate(self.recovery_references):
            ids = (self.recovery_reference_ids==i)&active
            self.recovery_phase[ids] = (self.recovery_phase[ids]+1).clamp_max(ref.num_frames-1)
        self.hold_steps[was_final&active] += 1
        self.hold_steps[~was_final] = 0
        self.just_reset.zero_()
        self._update_alignment_cache()

    def _update_metrics(self):
        self.metrics["target_error"] = self.recovery_target_error()


@configclass
class RecoveryOnlyCommandCfg(CommandTermCfg):
    class_type: type = RecoveryOnlyCommand
    asset_name: str = "robot"
    trajectory_file: str = MISSING
    supine_trajectory_file: str = MISSING
    prone_trajectory_file: str = MISSING
    urdf_path: str = MISSING
    anchor_body_name: str = "torso_link"
    body_names: list[str] = MISSING
    resampling_time_range: tuple[float,float] = (1.e9,1.e9)
    debug_vis: bool = False
    reset_perturbations: bool = True
    reset_pose_noise: dict = {"x":(-.02,.02),"y":(-.02,.02),"z":(0.,.02),
                              "roll":(-.03,.03),"pitch":(-.03,.03),"yaw":(-.05,.05)}
    reset_velocity_noise: dict = {"x":(-.1,.1),"y":(-.1,.1),"z":(-.05,.05),
                                  "roll":(-.1,.1),"pitch":(-.1,.1),"yaw":(-.1,.1)}
    # Symmetric uniform half-widths (rad, rad/s); defaults keep the original draws.
    reset_joint_position_noise: float = .03
    reset_joint_velocity_noise: float = .1
    floor_clearance: float = .01
    phase_bins: int = 20
    # Failure-adaptive look-ahead (BeyondMimic): a failure recorded at phase t also raises the bins
    # up to this many seconds before t (weight decay**seconds). 0: off, failures stay in their own bin.
    adaptive_kernel_seconds: float = 0.
    adaptive_kernel_decay: float = .8
    # Uniform share inside the adaptive branch (resets are already 50% uniform / 50% adaptive).
    # .2 (runs 14-21) makes the overall mix 60:40; 0 is StableMimic's 50:50.
    adaptive_inner_uniform: float = .2
    persistence_s: float = 2.
    grace_s: float = 1.
    upright_height: float = .65
    upright_alignment: float = math.cos(math.radians(30))
