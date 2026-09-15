"""Construct UMA and D3 calculators without keeping notebook-global state."""

from typing import Any

from uma_catalysis.structs.config import ModelConfig

VALID_UMA_TASKS = frozenset({"omat", "oc20"})


def load_predictor(config: ModelConfig) -> Any:
    """
    Load the configured FAIR Chemistry pretrained prediction unit.

    Parameters
    ----------
    config : uma_catalysis.structs.config.ModelConfig
        Model identifier used to load the UMA prediction unit.

    Returns
    -------
    Any
        FAIR Chemistry prediction unit compatible with ``FAIRChemCalculator``.
    """
    from fairchem.core import pretrained_mlip

    return pretrained_mlip.get_predict_unit(config.model_name)


def build_uma_calculator(predictor: Any, task_name: str) -> Any:
    """
    Construct a FAIR Chemistry calculator for one supported scientific task.

    Parameters
    ----------
    predictor : Any
        Prediction unit returned by :func:`load_predictor`.
    task_name : str
        UMA task identifier. ``"omat"`` is used for bulk and clean surfaces;
        ``"oc20"`` is used for adsorbate-surface chemistry.

    Returns
    -------
    Any
        Configured ``FAIRChemCalculator`` instance.

    Raises
    ------
    ValueError
        If ``task_name`` is not a supported UMA task.
    """
    if task_name not in VALID_UMA_TASKS:
        raise ValueError(
            f"task_name must be one of {sorted(VALID_UMA_TASKS)}, got {task_name!r}."
        )

    from fairchem.core import FAIRChemCalculator

    return FAIRChemCalculator(predictor, task_name=task_name)


def build_d3_calculator(config: ModelConfig) -> Any:
    """
    Construct the configured D3 endpoint-correction calculator.

    Parameters
    ----------
    config : uma_catalysis.structs.config.ModelConfig
        Device and damping settings for the D3 calculator.

    Returns
    -------
    Any
        Configured ``TorchDFTD3Calculator`` instance.
    """
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    return TorchDFTD3Calculator(device=config.d3_device, damping=config.d3_damping)
