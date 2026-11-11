"""CPU regression for the per-motion Recovery expert actor, without launching Kit.

AST loading isolates the actor classes from the package import. Run with the
existing RSL-RL5 runtime: /isaac-sim/python.sh -m pytest this_file.
"""
import ast
from pathlib import Path
import pytest
import torch
from tensordict import TensorDict
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories

SOURCE = Path(__file__).resolve().parents[3] / "cyclo_lab/manager_based/trace_recovery/policy.py"
DISTRIBUTION = {"class_name": "GaussianDistribution", "init_std": 1., "std_type": "log"}


def actors():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {"torch": torch, "nn": nn, "MLPModel": MLPModel, "MLP": MLP,
                 "unpad_trajectories": unpad_trajectories}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def observations(identities):
    n = len(identities)
    router = torch.nn.functional.one_hot(torch.tensor(identities), 2).float()
    return TensorDict({"policy": torch.randn(n, 508), "router": router}, batch_size=[n])


def build(name, obs):
    return actors()[name](obs, {"actor": ["policy"]}, "actor", 23, distribution_cfg=DISTRIBUTION)


def test_each_expert_matches_the_u14_recovery_expert_size():
    obs = observations([0, 1])
    split = build("RecoveryMotionExpertActor", obs)
    sizes = {k: sum(p.numel() for p in v.parameters()) for k, v in split.actor_experts.items()}
    assert sizes == {"supine": 427799, "prone": 427799}
    assert split.log_std.shape == (23,)


def test_routing_and_gradients_stay_within_the_owning_expert():
    torch.manual_seed(0)
    obs = observations([0, 1, 1, 0])
    split = build("RecoveryMotionExpertActor", obs)
    mean = split(obs)
    latent = split.get_latent(obs)
    for i, name in enumerate(("supine", "prone")):
        ids = obs["router"][:, i] > 0
        assert torch.allclose(mean[ids], split.actor_experts[name](latent[ids]))
    split.zero_grad()
    split(obs[obs["router"][:, 1] > 0]).sum().backward()  # Prone samples only.
    assert all(p.grad is None or not p.grad.any() for p in split.actor_experts["supine"].parameters())
    assert any(p.grad is not None and p.grad.any() for p in split.actor_experts["prone"].parameters())


def test_non_one_hot_router_is_rejected():
    obs = observations([0, 1])
    split = build("RecoveryMotionExpertActor", obs)
    obs["router"][0] = torch.tensor([.5, .5])
    with pytest.raises(ValueError):
        split(obs)
