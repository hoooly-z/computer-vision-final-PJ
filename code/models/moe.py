from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMoE(nn.Module):
    """Mixture-of-experts combiner，支持专家置信度输入。"""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        num_experts: int = 3,
        use_confidence: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.use_confidence = use_confidence
        gating_dim = num_experts * num_classes + (num_experts if use_confidence else 0)
        self.gating = nn.Sequential(
            nn.Linear(gating_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, expert_logits: torch.Tensor, confidences: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            expert_logits: Tensor (batch, num_experts, num_classes)
            confidences: Tensor (batch, num_experts) 或 None
        """
        batch = expert_logits.size(0)
        flat_logits = expert_logits.reshape(batch, -1)
        if self.use_confidence:
            if confidences is None:
                raise ValueError("Confidence tensor required when use_confidence=True")
            gate_input = torch.cat([flat_logits, confidences], dim=1)
        else:
            gate_input = flat_logits
        gate_scores = self.gating(gate_input)
        weights = F.softmax(gate_scores, dim=1).unsqueeze(-1)
        mixed = torch.sum(weights * expert_logits, dim=1)
        return mixed
