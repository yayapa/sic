# surv_icl_learning.py
#
# The transformer backbone (column embedding -> row interaction -> in-context
# learning) is adapted from TabICL (https://github.com/soda-inria/tabicl).
# The survival heads below (CoxPH / DeepHit / DeepEH) are SIC additions.
from __future__ import annotations
from typing import Dict
import torch
from torch import nn, Tensor
from .layers import Time_to_event_Embedding

from .encoders import Encoder


class SurvICLearning(nn.Module):
    """
    Minimal survival ICL head:
      R -> ICL encoder -> per-bin log2 hazards for QUERY rows.

    Assumptions:
    - You handle (E,T) -> (log_dt, is_event) and the loss in the trainer.
    - Per-dataset time bins are computed from SUPPORT events in the trainer.
    - No label injection, no hierarchical logic, no curves here.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_bins: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        surv_model_type: str = "deephit",
    ):
        super().__init__()
        self.d_model = d_model
        self.K = num_bins
        self.norm_first = norm_first
        self.surv_model_type = surv_model_type
        
        self.time_to_event_encoder = Time_to_event_Embedding(d_model, non_linear_time=True, num_events=2, ln=True)
        
        # ICL encoder over the whole (support + query) sequence.
        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        self.ln = nn.LayerNorm(d_model) #if norm_first else nn.Identity()
        
        if self.surv_model_type == "coxph":
            self.decoder_surv = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
            )
            self.surv_layer = nn.Linear(d_model*2, num_bins)
        elif self.surv_model_type == "deephit":
            self.decoder_surv = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
            )
            self.surv_layer = nn.Linear(d_model * 2, num_bins, bias=True)
        elif self.surv_model_type == "deepeh":
            self.decoder_surv = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
            )
            # DeepEH only consumes the first two columns (g1, g2); the extra
            # width is kept for checkpoint compatibility. See fit_predict_raw.
            self.surv_layer = nn.Linear(d_model * 2, num_bins, bias=True)
        else:
            raise ValueError(
                f"Unknown surv_model_type={self.surv_model_type!r}. "
                "Supported: 'coxph', 'deephit', 'deepeh'."
            )

    def forward(self, R: Tensor, y_train: Tensor, time_bin_grid: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        R : (B, T, d_model)
            Row embeddings with SUPPORT first, then QUERY.
        y_train : Tensor of shape (B,2, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data
        times_bin_grid : Tensor of shape (B, K)

        Returns
        -------
        dict with:
            "log_hazard": (B, T_query, K)  
        """
        train_size = y_train.shape[-1]
        
        # Encode training labels and add to R for SUPPORT rows.
        y_train_time = y_train[:, 0, :]
        y_train_event = y_train[:, 1, :]
        y_train_time_event_emb = self.time_to_event_encoder(y_train_time, y_train_event)
        R[:, :train_size] = R[:, :train_size] + y_train_time_event_emb  #+ y_train_event_enc + y_train_time_enc  + y_time_bin_enc
        src = self.tf_icl(R, attn_mask=train_size) # shape (B, T, d_model)
        src = self.ln(src)
        src = self.decoder_surv(src)
        src = self.surv_layer(src) # time dependent logits
        return {"log_hazard": src[:, train_size:, :], "log_hazard_train": src[:, :train_size, :]}