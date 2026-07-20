"""Agent factory — instantiate any agent type by name."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.ppo_agent import PPOAgent
from agents.dreamer_agent import DreamerAgent
from agents.worldmodel_agent import WorldModelAgent


def build_agent(
    agent_type: str,
    state_dim: int = 16,
    hyperparams: Optional[Dict[str, Any]] = None,
    use_mlp: bool = False,
    heterogeneous: bool = False,
    seed: int = 0,
):
    """
    Factory function returning an instantiated agent.

    Parameters
    ----------
    agent_type : 'ppo' | 'dreamer' | 'worldmodel'
    state_dim : int
    hyperparams : dict
    use_mlp : bool          PPO: use MLP policy (v1.2)
    heterogeneous : bool    WorldModel: mix predictor types (v1.2)
    seed : int
    """
    hp = hyperparams or {}
    atype = agent_type.lower()

    if atype == "ppo":
        return PPOAgent(state_dim=state_dim, hyperparams=hp, use_mlp=use_mlp, seed=seed)
    elif atype == "dreamer":
        return DreamerAgent(state_dim=state_dim, hyperparams=hp, seed=seed)
    elif atype == "worldmodel":
        return WorldModelAgent(
            state_dim=state_dim,
            hyperparams=hp,
            heterogeneous=heterogeneous,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")
