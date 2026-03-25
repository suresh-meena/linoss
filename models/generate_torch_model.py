"""Factory helpers for Torch-native experiment models."""

from __future__ import annotations


def create_torch_model(
    model_name: str,
    input_dim: int,
    label_dim: int,
    *,
    model_args: dict,
    classification: bool = True,
    output_step: int = 1,
):
    if model_name != "SLinOSS":
        raise ValueError(f"Unknown Torch model name: {model_name}")
    if classification and label_dim <= 1:
        raise ValueError("SLinOSS classification requires at least two output classes.")
    from models.SLinOSS import SLinOSS

    return SLinOSS(
        input_dim=input_dim,
        output_dim=label_dim,
        classification=classification,
        output_step=output_step,
        **model_args,
    )
