"""CPU regression for the paper-aligned unified Recovery actor, without launching Kit.

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


def actors():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {"torch": torch, "nn": nn, "MLPModel": MLPModel, "MLP": MLP,
                 "unpad_trajectories": unpad_trajectories}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def observations(n=8, scale=1.):
    return TensorDict({"policy": torch.randn(n, 508) * scale, "router": torch.zeros(n, 1)}, batch_size=[n])


def build(name, obs, std_type, normalization):
    distribution = {"class_name": "GaussianDistribution", "init_std": 1., "std_type": std_type}
    return actors()[name](obs, {"actor": ["policy"]}, "actor", 23,
                          obs_normalization=normalization, distribution_cfg=distribution)


def test_u14_contract_is_unchanged():
    obs = observations()
    with pytest.raises(ValueError):
        build("U14ProposalActor", obs, "log", True)
    with pytest.raises(ValueError):
        build("U14ProposalActor", obs, "scalar", False)
    assert build("U14ProposalActor", obs, "log", False).log_std.shape == (23,)


def test_paper_aligned_actor_normalizes_policy_only_and_uses_recovery_expert():
    torch.manual_seed(0)
    obs = observations(scale=5.)
    actor = build("PaperAlignedRecoveryActor", obs, "scalar", True)
    assert torch.allclose(actor.log_std, torch.zeros(23))  # init_std 1 in scalar space.
    actor.update_normalization(obs)
    assert actor.obs_normalizer.mean.shape[-1] == 508  # Router is not part of the input.
    calls = {"tracking": 0, "recovery": 0}
    for name in calls:
        actor.actor_experts[name].register_forward_hook(lambda *_a, key=name: calls.__setitem__(key, calls[key]+1))
    mean = actor(obs)
    assert mean.shape == (8, 23) and calls == {"tracking": 0, "recovery": 1}
    assert torch.allclose(mean, actor.actor_experts["recovery"](actor.get_latent(obs)))
