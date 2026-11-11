# SPDX-License-Identifier: Apache-2.0
"""Interactive exhibition: one K1 dances; visitors push it; it falls, gets up and dances on.

The integrated gated TRACE policy needs no mode switch from the environment: the actor sees only the live
Tracking reference and proprioception, and its proprioceptive gate picks the Tracking or Recovery expert. So the
robot simply runs on without resets (every termination is ignored except a NaN state) while the dance reference
keeps looping. Once a fallen robot lies still, the dance reference is turned to its lying heading
(trace_heading_alignment): Recovery was trained lying in one heading only, and without this a robot pushed
sideways rarely gets up (push test, model_40600: 15/44 -> 45/45 back to the dance).

Disturbances, in one of two modes (B toggles; the panel shows which):
  push mode          velocity impulse on the pelvis, like the training push event but stronger
  ball mode          a visible ball is thrown at the torso from the opposite side, so it knocks the robot the same
                     way; light / medium / strong = a soccer ball / a 3 kg ball / a 10 kg heavy ball
  I / K / J / L      knock the robot away from / toward the camera / to the left / to the right (screen directions)
  P                  random horizontal direction
  B                  push mode <-> ball mode
  1 / 2 / 3          strength: light / medium / strong
  right-click drag   pull a body with the mouse (Newton viewer picking)
  R                  reset: stand at the dance start (the only reset; a robot that stays down stays down)
  C                  camera follow on/off (mouse / WASD camera when off)
  G                  heading alignment on/off (for comparison; on by default)
The same actions are buttons in the "TRACE exhibition" window, which also shows the state and fall statistics.
--record writes the 3D view (without the panel) to session.mp4 in --output-dir, in simulation time.

Thrown balls are parked off-stage again after --ball-seconds, so none is left lying where the robot gets up.

Events (falls, recoveries, pushes, throws, resets) go to events.csv / summary.json in --output-dir.

usage (inside the container, /workspace/cyclo_lab):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/exhibition_trace_push.py \
    --eval-task Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-Play \
    --output-dir logs/eval/exhibition_$(date +%H%M%S) --checkpoint <run>/model_40600.pt
"""
import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

exhibition_parser = argparse.ArgumentParser(add_help=False)
exhibition_parser.add_argument('--output-dir', required=True)
exhibition_parser.add_argument('--eval-task', default='Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-Play')
exhibition_parser.add_argument('--push-speeds', type=float, nargs=3, default=(.5, 1.8, 3.),
                               help='light / medium / strong pelvis velocity change (m/s); training pushes are <= .5')
exhibition_parser.add_argument('--ball-speeds', type=float, nargs=3, default=(10., 10., 10.),
                               help='light / medium / strong ball launch speed (m/s)')
exhibition_parser.add_argument('--ball-seconds', type=float, default=3., help='a thrown ball is removed after this long')
exhibition_parser.add_argument('--balls', type=int, default=2, help='balls per strength level (rapid throws)')
exhibition_parser.add_argument('--mode', choices=('push', 'ball'), default='push', help='start mode')
exhibition_parser.add_argument('--record', action='store_true', help='record the 3D view to session.mp4 in --output-dir')
exhibition_parser.add_argument('--max-seconds', type=float, default=None, help='stop after this long (testing)')
exhibition_parser.add_argument('--script', default=None,
                               help='testing: "t:dir:level[:b],..." scheduled pushes (":b" = ball throw), e.g. "4:i:3,12:j:3:b"')
exhibition, remaining = exhibition_parser.parse_known_args()
output = Path(exhibition.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)  # never overwrite another session
sys.argv = [sys.argv[0], *remaining, '--task', exhibition.eval_task,
            '--num_envs', '1', '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import cv2
import imageio
import torch
import trace_heading_alignment
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor, refresh_reset_cache

UPRIGHT_PELVIS, FALLEN_PELVIS = .6, .45
UPRIGHT_ALIGNMENT = math.cos(math.radians(30))
RECORD_EVERY = 2  # steps per video frame (50 Hz control -> 25 fps)
RECOVERED_HOLD_S = 1.  # upright with the gate on Tracking this long = back to dancing
# Screen directions: the robot faces the camera, which sits on the -y side looking +y.
DIRECTIONS = {'i': (0., 1.), 'k': (0., -1.), 'j': (-1., 0.), 'l': (1., 0.)}
DIRECTION_NAMES = {'i': 'away', 'k': 'toward', 'j': 'left', 'l': 'right', 'p': 'random'}
LEVEL_NAMES = ('light', 'medium', 'strong')
# (mass kg, radius m, colour) per strength level: soccer ball, 3 kg ball, 10 kg heavy ball
BALL_TYPES = ((.45, .11, (.95, .95, .95)), (3., .13, (.95, .6, .1)), (10., .18, (.8, .15, .15)))
BALL_DISTANCE = 2.5  # launch point, m from the torso
PARK = (40., 40.)    # off-stage resting place of idle balls


def ball_name(level, k):
    return f'ball_{level}_{k}'


def add_balls(scene):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    for level, (mass, radius, color) in enumerate(BALL_TYPES):
        for k in range(exhibition.balls):
            setattr(scene, ball_name(level, k), RigidObjectCfg(
                prim_path=f'{{ENV_REGEX_NS}}/Ball_{level}_{k}',
                spawn=sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=park_position(level, k, radius))))


def park_position(level, k, radius):
    return (PARK[0]+level, PARK[1]+k, radius)


class ExhibitionCommand(TraceIntegratedCommand):
    """Every (re)start: Tracking at dance frame 0, standing on the reference."""
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_regimes[ids] = TRACKING
        super()._resample_command(ids)


original_run = play._run
holder = {}


def configured_run(env_cfg, agent_cfg):
    command = env_cfg.commands.reference_trajectory
    command.class_type = ExhibitionCommand
    command.reset_perturbations = False
    command.tracking_start_from_zero = True
    env_cfg.events.joint_home_offset_noise = None
    env_cfg.events.push_robot = None  # pushes come from the visitors
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.post_tracking_horizon_complete = None
    if getattr(env_cfg.terminations, 'tracking_origin_horizon_complete', None) is not None:
        env_cfg.terminations.tracking_origin_horizon_complete = None
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 1e6
    trace_heading_alignment.install(env_cfg, holder)
    add_balls(env_cfg.scene)  # not observed by the policy
    return original_run(env_cfg, agent_cfg)


class Recorder:
    """H.264 video of the viewer's 3D frames. One frame every RECORD_EVERY steps, so playback runs in simulation time;
    encoding happens on a thread (a full queue slows the show down rather than dropping frames)."""
    def __init__(self, viewer, path, fps):
        self.viewer, self.path, self.fps = viewer, path, fps
        self.frames, self.writer, self.size = 0, None, None
        self.queue = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._encode, daemon=True)

    def _encode(self):
        while (image := self.queue.get()) is not None:
            self.writer.append_data(image)

    def capture(self):
        image = self.viewer.get_frame().numpy()
        if self.writer is None:
            self.size = (image.shape[1] & ~1, image.shape[0] & ~1)  # H.264 needs even sides
            self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec='libx264', quality=8,
                                             pixelformat='yuv420p', macro_block_size=2)
            self.thread.start()
        if (image.shape[1], image.shape[0]) != self.size:  # window resized (or cropped to even sides)
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self.queue.put(image)
        self.frames += 1

    def close(self):
        if self.writer is None:
            return
        self.queue.put(None)
        self.thread.join()
        self.writer.close()
        self.writer = None
        print(f'[EXHIBITION] video: {self.path} ({self.frames} frames, {self.frames/self.fps:.1f} s)', flush=True)


class Show:
    """Push requests, fall/recovery bookkeeping and the on-screen panel."""
    def __init__(self, base, c):
        self.base, self.c, self.robot = base, c, c.robot
        self.level = 2
        self.ball_mode = exhibition.mode == 'ball'
        self.balls = [[base.scene[ball_name(level, k)] for k in range(exhibition.balls)] for level in range(3)]
        self.next_ball = [0, 0, 0]
        self.ball_until = {}  # (level, k) -> step after which the thrown ball is parked
        self.throws = 0
        self.ball_track = {}  # (level, k) -> hit statistics of the ball in flight
        self.follow = True
        self.requests = []  # (direction key, level, ball) queued by keys/buttons, applied before the next step
        self.reset_requested = False
        self.state = 'dancing'
        self.fall_step = None
        self.hold = 0
        self.step = 0
        self.falls = self.recoveries = self.pushes = 0
        self.recording = False
        self.recovery_times = []
        self.events = []
        self.gate_w_t = 1.
        self.pelvis = 0.
        self._keys_down = set()
        self.heading = holder['alignment'] = trace_heading_alignment.HeadingAlignment(c, base.step_dt)

    # --- input -----------------------------------------------------------------------
    def poll_keys(self, viewer):
        import pyglet
        keys = {'i': pyglet.window.key.I, 'k': pyglet.window.key.K, 'j': pyglet.window.key.J,
                'l': pyglet.window.key.L, 'p': pyglet.window.key.P, 'r': pyglet.window.key.R,
                'c': pyglet.window.key.C, 'g': pyglet.window.key.G, 'b': pyglet.window.key.B,
                '1': pyglet.window.key._1, '2': pyglet.window.key._2,
                '3': pyglet.window.key._3}
        down = {name for name, code in keys.items() if viewer.renderer.is_key_down(code)}
        for name in down-self._keys_down:  # act on the press edge only
            if name in DIRECTIONS or name == 'p':
                self.request(name)
            elif name == 'b':
                self.ball_mode = not self.ball_mode
            elif name in '123':
                self.level = int(name)-1
            elif name == 'r':
                self.reset_requested = True
            elif name == 'c':
                self.follow = not self.follow
            elif name == 'g':
                self.heading.enabled = not self.heading.enabled
        self._keys_down = down

    def request(self, key):
        self.requests.append((key, self.level, self.ball_mode))

    def panel(self, imgui):
        imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(330, 0), imgui.Cond_.first_use_ever)
        imgui.begin('TRACE exhibition')
        color = {'dancing': (.3, .9, .4), 'fallen': (1., .4, .3), 'returning': (1., .8, .2)}[self.state]
        label = {'dancing': 'DANCING', 'fallen': 'FALLEN - GETTING UP', 'returning': 'BACK UP - REJOINING THE DANCE'}
        imgui.text_colored(imgui.ImVec4(*color, 1.), label[self.state])
        imgui.text(f'pelvis {self.pelvis:.2f} m   gate (dance) {self.gate_w_t:.2f}')
        if self.state != 'dancing' and self.fall_step is not None:
            imgui.text(f'time since fall {(self.step-self.fall_step)*self.base.step_dt:.1f} s')
        imgui.separator()
        imgui.text('Disturbance (B toggles)')
        if imgui.radio_button('push', not self.ball_mode):
            self.ball_mode = False
        imgui.same_line()
        if imgui.radio_button('throw a ball', self.ball_mode):
            self.ball_mode = True
        imgui.text('Strength (1/2/3)')
        for i, name in enumerate(LEVEL_NAMES):
            mass = BALL_TYPES[i][0]
            detail = (f'{mass:g} kg ball, {exhibition.ball_speeds[i]:.0f} m/s' if self.ball_mode
                      else f'{exhibition.push_speeds[i]:.1f} m/s')
            if imgui.radio_button(f'{name} ({detail})', self.level == i):
                self.level = i
        imgui.text('Knock the robot (I/J/K/L, P random)')
        if imgui.button(' away (I) '):
            self.request('i')
        if imgui.button(' left (J) '):
            self.request('j')
        imgui.same_line()
        if imgui.button(' right (L) '):
            self.request('l')
        if imgui.button(' toward (K) '):
            self.request('k')
        imgui.same_line()
        if imgui.button(' random (P) '):
            self.request('p')
        imgui.text('or right-click a body and drag')
        imgui.separator()
        if imgui.button(' reset (R) '):
            self.reset_requested = True
        imgui.same_line()
        _, self.follow = imgui.checkbox('camera follow (C)', self.follow)
        _, self.heading.enabled = imgui.checkbox('heading alignment (G)', self.heading.enabled)
        imgui.text(f'dance heading turned {math.degrees(float(self.heading.offset[0])):.0f} deg')
        imgui.separator()
        imgui.text(f'pushes {self.pushes}   balls {self.throws}   falls {self.falls}   recovered {self.recoveries}')
        if self.recovery_times:
            imgui.text(f'last recovery {self.recovery_times[-1]:.1f} s   '
                       f'mean {sum(self.recovery_times)/len(self.recovery_times):.1f} s')
        if self.recording:
            imgui.text_colored(imgui.ImVec4(1., .3, .3, 1.), 'REC  session.mp4')
        imgui.end()

    # --- actions ---------------------------------------------------------------------
    def apply_pushes(self):
        while self.requests:
            key, level, ball = self.requests.pop(0)
            if key == 'p':
                angle = float(torch.rand(()))*2*math.pi
                direction = (math.cos(angle), math.sin(angle))
            else:
                direction = DIRECTIONS[key]
            if ball:
                self.throw(direction, level, DIRECTION_NAMES[key])
                continue
            speed = exhibition.push_speeds[level]
            velocity = tensor(self.robot.data.root_com_vel_w).clone()
            velocity[:, 0] += direction[0]*speed
            velocity[:, 1] += direction[1]*speed
            ids = torch.zeros(1, dtype=torch.int32, device=self.base.device)
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity.contiguous(), env_ids=ids)
            refresh_reset_cache(self.robot)
            self.pushes += 1
            self.log('push', direction=DIRECTION_NAMES[key], level=LEVEL_NAMES[level], speed=speed)

    def throw(self, direction, level, name):
        """Launch the next ball of this level from BALL_DISTANCE behind the torso (opposite to the knock direction),
        on a ballistic arc that reaches the torso centre; it knocks the robot along `direction`."""
        k = self.next_ball[level]
        self.next_ball[level] = (k+1) % exhibition.balls
        ball = self.balls[level][k]
        target = self.c.robot_anchor_pos_w[0].clone()
        speed = exhibition.ball_speeds[level]
        flight = BALL_DISTANCE/speed
        g = 9.81
        start = target.clone()
        start[0] -= direction[0]*BALL_DISTANCE
        start[1] -= direction[1]*BALL_DISTANCE
        dev = self.base.device
        pose = torch.tensor([[float(start[0]), float(start[1]), float(start[2]), 0., 0., 0., 1.]], device=dev)
        velocity = torch.tensor([[direction[0]*speed, direction[1]*speed, .5*g*flight, 0., 0., 0.]], device=dev)
        ids = torch.zeros(1, dtype=torch.int32, device=dev)
        ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        ball.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
        self.ball_until[(level, k)] = self.step+round(exhibition.ball_seconds/self.base.step_dt)
        self.ball_track[(level, k)] = {'min_gap_m': 9., 'speed_in': speed, 'speed_after': None,
                                       'pelvis_speed_before': float(tensor(self.robot.data.root_com_vel_w)[0, :2].norm()),
                                       'pelvis_speed_peak': 0.}
        self.throws += 1
        self.log('throw', direction=name, level=LEVEL_NAMES[level], speed=speed, ball_kg=BALL_TYPES[level][0])

    def track_balls(self):
        """Closest approach of each ball in flight to the robot's bodies (surface gap ~ centre distance - radius)."""
        bodies = self.c.robot_body_pos_w[0]
        pelvis_speed = float(tensor(self.robot.data.root_com_vel_w)[0, :2].norm())
        for (level, k), t in self.ball_track.items():
            ball = self.balls[level][k]
            centre = tensor(ball.data.root_pos_w)[0]
            gap = float((bodies-centre).norm(dim=-1).min())-BALL_TYPES[level][1]
            t['min_gap_m'] = min(t['min_gap_m'], gap)
            t['pelvis_speed_peak'] = max(t['pelvis_speed_peak'], pelvis_speed)
            if self.ball_until[(level, k)]-self.step == round(exhibition.ball_seconds/self.base.step_dt)-round(.5/self.base.step_dt):
                t['speed_after'] = float(tensor(ball.data.root_com_vel_w)[0, :3].norm())  # 0.5 s after the throw

    def park_balls(self, everything=False):
        dev = self.base.device
        ids = torch.zeros(1, dtype=torch.int32, device=dev)
        for (level, k), until in list(self.ball_until.items()):
            if everything or self.step >= until:
                t = self.ball_track.pop((level, k), None)
                if t is not None:
                    self.log('ball_result', level=LEVEL_NAMES[level], **{key: (round(v, 3) if v is not None else None)
                                                                         for key, v in t.items()})
                x, y, z = park_position(level, k, BALL_TYPES[level][1])
                ball = self.balls[level][k]
                ball.write_root_pose_to_sim_index(root_pose=torch.tensor([[x, y, z, 0., 0., 0., 1.]], device=dev), env_ids=ids)
                ball.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev), env_ids=ids)
                del self.ball_until[(level, k)]

    def log(self, kind, **data):
        row = {'step': self.step, 'time_s': round(self.step*self.base.step_dt, 2), 'event': kind,
               'pelvis_m': round(self.pelvis, 3), 'gate_w_t': round(self.gate_w_t, 3), **data}
        self.events.append(row)
        print('[EXHIBITION]', json.dumps(row), flush=True)

    def update(self, gate_w_t):
        """Fall -> getting up -> back to the dance, from pelvis height, tilt and the gate."""
        self.gate_w_t = gate_w_t
        self.pelvis = float(self.c.robot_body_pos_w[0, 0, 2])
        alignment = float(-tensor(self.robot.data.projected_gravity_b)[0, 2])
        upright = self.pelvis >= UPRIGHT_PELVIS and alignment >= UPRIGHT_ALIGNMENT
        if self.state == 'dancing':
            self.heading.update(torch.tensor([False], device=self.base.device))  # keeps the settle counter at 0
            if self.pelvis < FALLEN_PELVIS:
                self.state, self.fall_step, self.hold = 'fallen', self.step, 0
                self.falls += 1
                self.log('fall')
            return
        if upright and self.state == 'fallen':
            self.state = 'returning'
            self.log('stood_up', seconds=round((self.step-self.fall_step)*self.base.step_dt, 2))
        if self.state == 'returning':
            if not upright:
                self.state = 'fallen' if self.pelvis < FALLEN_PELVIS else 'returning'
            self.hold = self.hold+1 if (upright and gate_w_t >= .9) else 0
            if self.hold*self.base.step_dt >= RECOVERED_HOLD_S:
                seconds = (self.step-self.fall_step)*self.base.step_dt
                self.recovery_times.append(seconds)
                self.recoveries += 1
                self.state, self.fall_step = 'dancing', None
                self.log('back_to_dance', seconds=round(seconds, 2))
                return
        locked = self.heading.update(torch.tensor([True], device=self.base.device))
        if bool(locked[0]):
            self.log('heading_aligned', offset_deg=round(math.degrees(float(self.heading.offset[0])), 1))

    def reset_done(self):
        self.state, self.fall_step, self.hold = 'dancing', None, 0
        self.heading.reset()
        self.requests.clear()
        self.park_balls(everything=True)
        self.log('reset')

    def camera(self, visualizer, smooth=.1):
        if not self.follow:
            return
        target = self.c.robot_body_pos_w[0, 0].cpu()
        goal = torch.tensor([float(target[0]), float(target[1]), .55])
        self._cam = goal if getattr(self, '_cam', None) is None else self._cam+(goal-self._cam)*smooth
        t = self._cam
        visualizer._apply_camera_pose(((float(t[0]), float(t[1])-4.2, 1.6), (float(t[0]), float(t[1]), float(t[2]))))


def scripted_pushes():
    out = {}
    for item in filter(None, (exhibition.script or '').split(',')):
        t, key, level = item.split(':')[:3]
        out[round(float(t)/.02)] = (key, int(level)-1, item.endswith(':b'))
    return out


def run_show(env, policy, mouse_drag_viewer, newton_state, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    assert c.regimes.tolist() == [TRACKING]
    manager = base.termination_manager
    original_compute = manager.compute

    def nan_only_compute():  # every failure/time-out is computed but ignored; a NaN state still resets
        done = original_compute()
        keep = (manager.get_term('nan_detected').clone() if 'nan_detected' in manager.active_terms
                else torch.zeros_like(done))
        manager._terminated_buf &= keep
        manager._truncated_buf &= keep
        return done & keep
    manager.compute = nan_only_compute

    show = Show(base, c)
    visualizer = viewer = None
    if base.sim.visualizers:
        viewer, newton_state = play._resolve_newton_mouse_drag(env)  # right-click drag picking
        visualizer = next(v for v in base.sim.visualizers if getattr(v, '_viewer', None) is viewer)
        viewer.register_ui_callback(show.panel, 'free')
        print('[EXHIBITION] I/J/K/L knock, P random, B push/ball mode, 1/2/3 strength, R reset, C camera follow, '
              'right-click drag to pull', flush=True)
    recorder = None
    if exhibition.record:
        if viewer is None:
            raise RuntimeError('--record needs the Newton viewer window (do not use --headless)')
        recorder = Recorder(viewer, output/'session.mp4', round(1./(RECORD_EVERY*base.step_dt)))
        show.recording = True
    script = scripted_pushes()
    limit = None if exhibition.max_seconds is None else round(exhibition.max_seconds/base.step_dt)
    obs = env.get_observations()
    show.log('start', checkpoint=play.args_cli.checkpoint)
    try:
        with torch.inference_mode():
            while play.simulation_app.is_running() and all(v.is_running() for v in base.sim.visualizers):
                if limit is not None and show.step >= limit:
                    break
                start = time.monotonic()
                if viewer is not None:
                    show.poll_keys(viewer)
                if show.step in script:
                    show.requests.append(script[show.step])
                if show.reset_requested:
                    show.reset_requested = False
                    env.reset()
                    obs = env.get_observations()
                    policy.reset(torch.ones(1, dtype=torch.bool, device=base.device))
                    show.reset_done()
                action = policy(obs)
                gate = getattr(policy, 'last_gate', None)
                gate_w_t = 1. if gate is None else float(gate['weights'][0, 0])
                show.apply_pushes()
                show.track_balls()
                show.park_balls()
                if viewer is not None:
                    viewer.apply_forces(newton_state())
                obs, _, done, _ = env.step(action)
                policy.reset(done)
                if bool(done[0]):
                    show.log('nan_reset')
                    show.reset_done()
                show.step += 1
                show.update(gate_w_t)
                if visualizer is not None:
                    show.camera(visualizer)
                if recorder is not None and show.step % RECORD_EVERY == 0:
                    recorder.capture()
                if visualizer is not None:  # real time on screen; tests run as fast as possible
                    time.sleep(max(0., base.step_dt-(time.monotonic()-start)))
    finally:  # also if the loop raises: an unclosed mp4 loses its last frames
        if recorder is not None:
            recorder.close()
        with (output/'events.csv').open('w', newline='') as stream:
            fields = sorted({k for row in show.events for k in row}, key=lambda k: (k not in ('step', 'time_s', 'event'), k))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(show.events)
        summary = {'checkpoint': play.args_cli.checkpoint, 'seconds': round(show.step*base.step_dt, 2),
                   'pushes': show.pushes, 'throws': show.throws, 'falls': show.falls, 'recoveries': show.recoveries,
                   'recovery_seconds': [round(s, 2) for s in show.recovery_times],
                   'push_speeds': list(exhibition.push_speeds), 'ball_speeds': list(exhibition.ball_speeds),
                   'ball_kg': [b[0] for b in BALL_TYPES], 'final_state': show.state}
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
        print('[EXHIBITION-RESULT]', json.dumps(summary), flush=True)
    env.close()


play._run = configured_run
play._play = run_show
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
