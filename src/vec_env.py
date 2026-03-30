"""Vectorized environment using multiprocessing for parallel game simulation.

Runs N game environments in separate processes, collecting observations
and stepping all environments in parallel. This is the single biggest
speedup for CPU-based training.
"""

from __future__ import annotations
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import Optional

import numpy as np


def _worker(pipe, env_fn):
    """Worker process that runs a single environment."""
    env = env_fn()
    while True:
        cmd, data = pipe.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                # Auto-reset
                final_info = info.copy()
                obs, info = env.reset()
                final_info["_final_obs"] = True
            else:
                final_info = info
            pipe.send((obs, reward, terminated, truncated, final_info))
        elif cmd == "reset":
            obs, info = env.reset()
            pipe.send((obs, info))
        elif cmd == "close":
            env.close()
            pipe.close()
            break
        elif cmd == "get_spaces":
            pipe.send((env.observation_space, env.action_space, getattr(env, 'action_dim', None)))


class VecEnv:
    """Vectorized environment running N envs in parallel processes."""

    def __init__(self, env_fns: list, start_method: str = "fork"):
        self.num_envs = len(env_fns)
        self.waiting = False

        ctx = mp.get_context(start_method)
        self.parent_pipes = []
        self.procs = []

        for env_fn in env_fns:
            parent_pipe, child_pipe = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child_pipe, env_fn), daemon=True)
            proc.start()
            child_pipe.close()
            self.parent_pipes.append(parent_pipe)
            self.procs.append(proc)

        # Get spaces from first env
        self.parent_pipes[0].send(("get_spaces", None))
        self.observation_space, self.action_space, self.action_dim = self.parent_pipes[0].recv()

    def reset(self):
        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        results = [pipe.recv() for pipe in self.parent_pipes]
        obs = np.array([r[0] for r in results])
        infos = [r[1] for r in results]
        return obs, infos

    def step(self, actions):
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        results = [pipe.recv() for pipe in self.parent_pipes]
        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminateds = np.array([r[2] for r in results])
        truncateds = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for proc in self.procs:
            proc.join(timeout=5)


def make_vec_env(env_class, num_envs: int, **kwargs):
    """Create a vectorized environment with N parallel workers."""
    def make_env(seed):
        def _init():
            return env_class(seed=seed, **kwargs)
        return _init

    env_fns = [make_env(i) for i in range(num_envs)]
    return VecEnv(env_fns)
