from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    num_pids: int
    num_states: int
    pid_emb: int = 32
    state_emb: int = 8
    cont_dim: int = 2
    hidden: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class LSTMNextPid(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg

        self.pid_emb = nn.Embedding(cfg.num_pids, cfg.pid_emb)
        self.state_emb = nn.Embedding(cfg.num_states, cfg.state_emb)

        in_feat = cfg.pid_emb + cfg.state_emb + cfg.cont_dim

        self.lstm = nn.LSTM(
            input_size=in_feat,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=(cfg.dropout if cfg.num_layers > 1 else 0.0),
        )

        self.head = nn.Linear(cfg.hidden, cfg.num_pids)

    def forward(
        self,
        pid: torch.Tensor,
        cont: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if state is None:
            raise ValueError("state tensor is required")

        pid_e = self.pid_emb(pid)      # (B, L, pid_emb)
        st_e = self.state_emb(state)   # (B, L, state_emb)

        x = torch.cat([pid_e, st_e, cont], dim=-1)   # (B, L, in_feat)
        out, _ = self.lstm(x)                        # (B, L, hidden)

        last = out[:, -1, :]                         # (B, hidden)
        logits = self.head(last)                     # (B, num_pids)
        return logits