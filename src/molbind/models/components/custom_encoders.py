from pathlib import Path
from typing import (
    Callable,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.dimenet import (
    DimeNet,
    OutputBlock,
    triplets,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter
from transformers import AutoConfig, AutoModel, RobertaForCausalLM
from transformers import BertModel, BertTokenizer

from molbind.models.components.base_encoder import (
    BaseModalityEncoder,
)
from molbind.utils import rename_keys_with_prefix, select_device


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("ibm/MoLFormer-XL-both-10pct", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self):
        self.encoder = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        token_ids, attention_mask = x if len(x) == 2 else (x[0], x[1])
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        return output.pooler_output


class PolymerNameEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("FacebookAI/roberta-base", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self) -> None:
        config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = RobertaForCausalLM.from_pretrained(self.model_name, config=config)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


class PsmilesEncoder(BaseModalityEncoder):
    """Encoder for Polymer SMILES (PSMILES) using PolyBERT model."""

    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("kuelumbus/polyBERT", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self):
        """Initialize the PolyBERT encoder for PSMILES."""
        self.encoder = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass for PSMILES encoding.

        Args:
            x: Tuple of (token_ids, attention_mask)

        Returns:
            Encoded PSMILES representation
        """
        token_ids, attention_mask = x if len(x) == 2 else (x[0], x[1])
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        # PolyBERT uses mean pooling over token dimension
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BigsmilesEncoder(BaseModalityEncoder):
    """Encoder for BigSMILES using ChemBERTa-zinc model."""

    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("seyonec/ChemBERTa-zinc-base-v1", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self):
        """Initialize the ChemBERTa encoder for BigSMILES."""
        self.encoder = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass for BigSMILES encoding.

        Args:
            x: Tuple of (token_ids, attention_mask)

        Returns:
            Encoded BigSMILES representation
        """
        token_ids, attention_mask = x if len(x) == 2 else (x[0], x[1])
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        # Use pooler output if available, otherwise mean pooling
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        else:
            # Mean pooling as fallback
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
