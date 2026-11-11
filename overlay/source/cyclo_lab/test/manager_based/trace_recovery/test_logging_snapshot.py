"""CPU regression for the actual TRACE logging helpers, without launching Kit.

AST loading isolates the pure helpers from the simulator import. Run with
the existing RSL-RL5 runtime: /isaac-sim/python.sh -m pytest this_file.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from rsl_rl.utils.logger import Logger

SOURCE = Path(__file__).resolve().parents[3] / "cyclo_lab/manager_based/trace_recovery/environment.py"


def helpers():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_snapshot_log_value", "_new_step_log"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, key, value, step):
        self.values[key] = float(value)


def reconstruct(snapshot):
    cfg = {"algorithm": {"rnd_cfg": None}, "num_steps_per_env": 3, "run_name": None}
    logger = Logger(None, cfg, {}, 4096, False, 1, 0, "cpu")
    logger.writer, logger.logger_type = Writer(), "tensorboard"
    previous, references = {}, []
    new_log = helpers()["_new_step_log"] if snapshot else lambda log: log
    for count in (10, 0, 1):
        current = new_log(previous)
        current["Trace/test_event_fraction"] = torch.tensor(count / 4096)
        logger.process_env_step(torch.zeros(4096), torch.zeros(4096, dtype=torch.long), {"log": current})
        references.append(current)
        previous = current
    unique = len({id(log) for log in references})
    logger.log(it=0, start_it=0, total_it=1, collect_time=1., learn_time=1., loss_dict={},
               learning_rate=.001, action_std=torch.ones(23), rnd_weight=None, print_minimal=True)
    return round(logger.writer.values["Trace/test_event_fraction"] * 4096 * 3), unique


def test_native_logger_event_reconstruction_is_eleven():
    assert reconstruct(True) == (11, 3)
    # The negative control must fail the eleven-event contract if reuse returns.
    assert reconstruct(False) != (11, 3)


def test_actual_environment_step_keeps_eleven_events_and_distinct_objects():
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    class Base:
        def step(self, action):
            return {}, torch.zeros(64), torch.zeros(64, dtype=torch.bool), torch.zeros(64, dtype=torch.bool), self.extras
    namespace = {"torch": torch, "ManagerBasedRLEnv": Base}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    c = SimpleNamespace(
        recovery_reference_ids=torch.arange(64) % 2, recovery_phase=torch.zeros(64, dtype=torch.long),
        phase_bins=lambda identity, phase: torch.zeros(64, dtype=torch.long),
        final_mask=lambda: torch.zeros(64, dtype=torch.bool), upright_mask=lambda: torch.zeros(64, dtype=torch.bool),
        step_error_snapshot={"body_position": torch.ones(64)},
        adaptive_bin_probabilities=lambda identity: torch.ones(20) / 20,
        failure_score_by_reference_bin=torch.zeros(2, 20), pending_events={"test": torch.tensor(0)},
        reset_bins=torch.zeros(2, 20, dtype=torch.long), failure_onset_bins=torch.zeros(2, 20, dtype=torch.long),
        failure_detection_bins=torch.zeros(2, 20, dtype=torch.long))
    env = object.__new__(namespace["TraceRecoveryEnv"])
    env.num_envs, env.device, env.extras, env._logged_histograms = 64, "cpu", {}, {}
    env.command_manager = SimpleNamespace(get_term=lambda name: c)
    cfg = {"algorithm": {"rnd_cfg": None}, "num_steps_per_env": 3, "run_name": None}
    logger = Logger(None, cfg, {}, 64, False, 1, 0, "cpu")
    logger.writer, logger.logger_type = Writer(), "tensorboard"
    for count in (10, 0, 1):
        c.pending_events["test"].fill_(count)
        _, reward, terminated, truncated, extras = env.step(torch.zeros(64, 23))
        logger.process_env_step(reward, terminated | truncated, extras)
    assert len({id(log) for log in logger.ep_extras}) == 3
    logger.log(it=0, start_it=0, total_it=1, collect_time=1., learn_time=1., loss_dict={},
               learning_rate=.001, action_std=torch.ones(23), rnd_weight=None, print_minimal=True)
    assert round(logger.writer.values["Trace/test_event_fraction"] * 64 * 3) == 11


def test_nested_native_values_are_independent_and_no_large_state_copy():
    snapshot = helpers()["_snapshot_log_value"]
    original = {"nested": [torch.tensor(3.), {"small": torch.ones(2)}]}
    cloned = snapshot(original)
    original["nested"][0].zero_()
    original["nested"][1]["small"].zero_()
    assert cloned["nested"][0].item() == 3
    assert torch.equal(cloned["nested"][1]["small"], torch.ones(2))
    assert original is not cloned and original["nested"] is not cloned["nested"]
    try:
        snapshot(torch.ones(4096, 508))
    except ValueError:
        pass
    else:
        raise AssertionError("A batched policy/state tensor must not be copied into logging")


def test_explicit_reset_moves_only_logging_baselines():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TraceRecoveryEnv")
    reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    fake_cls = ast.ClassDef(name="TraceRecoveryEnv", bases=[ast.Name(id="Base", ctx=ast.Load())],
                           keywords=[], body=[reset], decorator_list=[])
    ast.fix_missing_locations(fake_cls)
    token = object()
    class Base:
        def reset(self, *args, **kwargs):
            return token
    c = SimpleNamespace(pending_events={"reset": torch.tensor(5)}, reset_bins=torch.ones(2, 20),
                        failure_onset_bins=torch.ones(2, 20), failure_detection_bins=torch.ones(2, 20),
                        events={"reset": torch.tensor(5)}, failure_score_by_reference_bin=torch.ones(2, 20))
    namespace = {"Base": Base}
    exec(compile(ast.Module(body=[fake_cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
    env = namespace["TraceRecoveryEnv"]()
    env.command_manager = SimpleNamespace(get_term=lambda name: c)
    assert env.reset() is token
    assert c.pending_events["reset"].item() == 0 and c.events["reset"].item() == 5
    assert torch.equal(c.failure_score_by_reference_bin, torch.ones(2, 20))
    c.reset_bins.zero_()
    assert torch.equal(env._logged_histograms["reset_bins"], torch.ones(2, 20))
