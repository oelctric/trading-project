"""Deep Q-Network agent (PyTorch) for ScalperEnv.

Compact, self-contained DQN with a target network and experience replay.
Tensor shapes are annotated. The whole point of the sandbox is to see what
policy emerges: if trading has negative expectancy after 8 bps friction, the
Q-values for actions 1/2 stay below action 0 and the greedy policy collapses
to Hold. If the agent finds positive-EV state pockets, it will trade there.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, state) -> (B, A)
        return self.net(x)


class Replay:
    def __init__(self, cap: int = 50_000):
        self.buf: deque = deque(maxlen=cap)

    def push(self, *tr):                       # (s, a, r, s2, done)
        self.buf.append(tr)

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.as_tensor(s, dtype=torch.float32),
                torch.as_tensor(a, dtype=torch.long),
                torch.as_tensor(r, dtype=torch.float32),
                torch.as_tensor(s2, dtype=torch.float32),
                torch.as_tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 device="cpu"):
        self.device = torch.device(device)
        self.q = QNet(state_dim, n_actions).to(self.device)
        self.tgt = QNet(state_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.n_actions = gamma, n_actions
        self.replay = Replay()
        self.eps = 1.0

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)  # (1, state)
            return int(self.q(s).argmax(1).item())

    def learn(self, batch_size=256):
        if len(self.replay) < batch_size:
            return None
        s, a, r, s2, d = (t.to(self.device) for t in
                          self.replay.sample(batch_size))
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)          # (B,)
        with torch.no_grad():
            q_next = self.tgt(s2).max(1).values                    # (B,)
            target = r + self.gamma * q_next * (1 - d)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.tgt.load_state_dict(self.q.state_dict())
