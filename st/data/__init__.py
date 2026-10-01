"""Synthetic tasks and memory-mapped token data (public exports)."""
from .synthetic import (
    VOCAB, KEY, P, Q, SEP, FILL0, FILL1, MQAR_KEYS, MQAR_FILL0, MQAR_FILL1,
    passkey_batch, copying_batch, mqar_batch, resolve_npairs,
)
from .tokens import TokenDataset, enwik8_dataset
