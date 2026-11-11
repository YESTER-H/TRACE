# SPDX-License-Identifier: Apache-2.0
"""Policy-step exact logging, outside actor/critic/reward and reset means."""
import torch
from isaaclab.envs import ManagerBasedRLEnv


def _snapshot_log_value(value):
    """Copy only small logging values; never copy a batched robot/rollout tensor."""
    if isinstance(value, torch.Tensor):
        if value.numel() > 64:
            raise ValueError("TRACE log snapshots require scalar/small tensors, not batched state")
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_log_value(item) for item in value)
    return value


def _new_step_log(previous):
    """Native reset metrics are snapshotted; every Trace/* scalar is rebuilt below."""
    return {key: _snapshot_log_value(value) for key, value in previous.items()
            if not key.startswith("Trace/")}


class TraceRecoveryEnv(ManagerBasedRLEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._logged_histograms = {}

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        # Explicit/setup reset is outside a PPO rollout. Raw simulator counters,
        # failure EMA and all MDP state remain untouched; only logging baselines move.
        c = self.command_manager.get_term("reference_trajectory")
        for value in c.pending_events.values():
            value.zero_()
        self._logged_histograms = {name: getattr(c, name).clone() for name in
                                   ("reset_bins", "failure_onset_bins", "failure_detection_bins")}
        return output

    def step(self, action):
        c = self.command_manager.get_term("reference_trajectory")
        # Identity/phase which generated this PPO action, before autoreset.
        identity,phase = c.recovery_reference_ids.clone(),c.recovery_phase.clone()
        bins = c.phase_bins(identity,phase)
        final,upright = c.final_mask().clone(),c.upright_mask().clone()
        output = super().step(action)
        # RSL-RL buffers dict references until iteration aggregation. A fresh
        # dict is required even when the native step did not reset any env.
        log = _new_step_log(self.extras.get("log", {}))
        reward = output[1]
        for i,name in enumerate(("supine","prone")):
            mask = identity==i
            log[f"Trace/{name}_mode_step_fraction"] = mask.float().mean()
            log[f"Trace/{name}_reward_contribution"] = (reward*mask).mean()
            for key,value in c.step_error_snapshot.items():
                log[f"Trace/{name}_{key}_error_contribution"] = (value*mask).mean()
            log[f"Trace/{name}_final_frame_hold_step_fraction"] = (mask&final).float().mean()
            log[f"Trace/{name}_final_frame_not_upright_step_fraction"] = (mask&final&~upright).float().mean()
            for b in range(20):
                prefix = f"Trace/{name}_phase_bin_{b:02d}"
                log[prefix+"_step_fraction"] = (mask&(bins==b)).float().mean()
                log[prefix+"_adaptive_probability"] = c.adaptive_bin_probabilities(i)[b].clone()
                log[prefix+"_failure_ema"] = c.failure_score_by_reference_bin[i,b].clone()
        log["Trace/tracking_mode_step_fraction"] = torch.zeros((),device=self.device)
        log["Trace/supine_failure_window_step_fraction"] = ((identity==0)&(phase>=120)&(phase<=180)).float().mean()
        for name,value in c.pending_events.items():
            log[f"Trace/{name}_event_fraction"] = value.float().clone()/self.num_envs
            value.zero_()
        for attr in ("reset_bins","failure_onset_bins","failure_detection_bins"):
            current = getattr(c,attr).clone()
            previous = self._logged_histograms.get(attr,torch.zeros_like(current))
            delta = current-previous
            self._logged_histograms[attr] = current
            for i,name in enumerate(("supine","prone")):
                for b in range(20):
                    log[f"Trace/{name}_phase_bin_{b:02d}_{attr}_event_fraction"] = delta[i,b].float()/self.num_envs
        self.extras["log"] = log
        return output


class TraceIntegratedEnv(TraceRecoveryEnv):
    """Recovery-only tags kept (now masked to the Recovery regime) plus U1.5 regime/handoff tags."""
    HISTOGRAMS = ("reset_bins", "failure_onset_bins", "failure_detection_bins")
    TRACKING_HISTOGRAMS = ("tracking_reset_bins", "tracking_failure_bins")

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        c = self.command_manager.get_term("reference_trajectory")
        for name in self.TRACKING_HISTOGRAMS:
            self._logged_histograms[name] = getattr(c, name).clone()
        return output

    def _delta(self, c, attr):
        current = getattr(c,attr).clone()
        delta = current-self._logged_histograms.get(attr,torch.zeros_like(current))
        self._logged_histograms[attr] = current
        return delta.float()/self.num_envs

    def step(self, action):
        from .integrated import TRACKING, RECOVERY, TRANSITION
        c = self.command_manager.get_term("reference_trajectory")
        # Regime/identity/phase which generated this PPO action, before autoreset.
        regimes,origins = c.regimes.clone(),c.episode_origins.clone()
        identity,phase = c.recovery_reference_ids.clone(),c.recovery_phase.clone()
        post = c.post_tracking_active.clone()&(regimes==TRACKING)
        bins = c.phase_bins(identity,phase)
        tracking_bins = c.tracking_bins(c.tracking_phase)
        final,upright = c.final_mask().clone(),c.upright_mask().clone()
        alpha = c.transition_alpha.clone()
        output = ManagerBasedRLEnv.step(self, action)
        log = _new_step_log(self.extras.get("log", {}))
        reward = output[1]
        for mode,name in ((TRACKING,"tracking"),(RECOVERY,"recovery"),(TRANSITION,"transition")):
            mask = regimes==mode
            log[f"Trace/{name}_mode_step_fraction"] = mask.float().mean()
            log[f"Trace/{name}_reward_contribution"] = (reward*mask).mean()
        log["Trace/post_tracking_step_fraction"] = post.float().mean()
        log["Trace/tracking_origin_step_fraction"] = (origins==TRACKING).float().mean()
        log["Trace/transition_alpha_masked_mean"] = (alpha*(regimes==TRANSITION)).mean()
        recovery = regimes==RECOVERY
        for i,name in enumerate(("supine","prone")):
            mask = recovery&(identity==i)
            log[f"Trace/{name}_mode_step_fraction"] = mask.float().mean()
            log[f"Trace/{name}_reward_contribution"] = (reward*mask).mean()
            for key,value in c.step_error_snapshot.items():
                log[f"Trace/{name}_{key}_error_contribution"] = (value*mask).mean()
            log[f"Trace/{name}_final_frame_hold_step_fraction"] = (mask&final).float().mean()
            log[f"Trace/{name}_final_frame_not_upright_step_fraction"] = (mask&final&~upright).float().mean()
            for b in range(20):
                prefix = f"Trace/{name}_phase_bin_{b:02d}"
                log[prefix+"_step_fraction"] = (mask&(bins==b)).float().mean()
                log[prefix+"_adaptive_probability"] = c.adaptive_bin_probabilities(i)[b].clone()
                log[prefix+"_failure_ema"] = c.failure_score_by_reference_bin[i,b].clone()
        probabilities = c.tracking_adaptive_probabilities()
        for b in range(20):
            prefix = f"Trace/tracking_bin_{b:02d}"
            log[prefix+"_step_fraction"] = ((regimes==TRACKING)&(tracking_bins==b)).float().mean()
            log[prefix+"_adaptive_probability"] = probabilities[b].clone()
            log[prefix+"_failure_score"] = c.tracking_failure_scores[b].clone()
        for name,value in c.pending_events.items():
            log[f"Trace/{name}_event_fraction"] = value.float().clone()/self.num_envs
            value.zero_()
        for attr in self.HISTOGRAMS:
            delta = self._delta(c,attr)
            for i,name in enumerate(("supine","prone")):
                for b in range(20):
                    log[f"Trace/{name}_phase_bin_{b:02d}_{attr}_event_fraction"] = delta[i,b]
        for attr in self.TRACKING_HISTOGRAMS:
            delta = self._delta(c,attr)
            for b in range(20):
                log[f"Trace/tracking_bin_{b:02d}_{attr}_event_fraction"] = delta[b]
        self.extras["log"] = log
        return output
