"""Factory helpers for Torch-native experiment models."""

from __future__ import annotations


def create_torch_model(
    model_name: str,
    input_dim: int,
    label_dim: int,
    *,
    model_args: dict,
):
    if model_name != "SLinOSS":
        raise ValueError(f"Unknown Torch model name: {model_name}")
    if label_dim <= 1:
        raise ValueError(
            "SLinOSSClassifier is classification-only and requires at least two classes."
        )
    from models.SLinOSS import SLinOSSClassifier

    return SLinOSSClassifier(
        input_dim=input_dim,
        num_classes=label_dim,
        **model_args,
    )
