import pytest
import torch
from torch import nn

from when_debate_helps.perturbation import GaussianPerturber


def test_perturbation_is_deterministic_and_restores_exactly() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    perturber = GaussianPerturber(model, verify_restore=True)

    with perturber.perturb(seed=17, sigma=0.1):
        first = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        assert any(not torch.equal(first[name], originals[name]) for name in originals)

    assert all(torch.equal(parameter, originals[name]) for name, parameter in model.named_parameters())

    with perturber.perturb(seed=17, sigma=0.1):
        second = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    assert all(torch.equal(first[name], second[name]) for name in first)
    assert all(torch.equal(parameter, originals[name]) for name, parameter in model.named_parameters())


def test_restore_runs_when_evaluation_raises() -> None:
    model = nn.Linear(2, 2)
    original = model.weight.detach().clone()
    perturber = GaussianPerturber(model)
    with pytest.raises(RuntimeError, match="synthetic failure"), perturber.perturb(seed=3, sigma=0.2):
        raise RuntimeError("synthetic failure")
    assert torch.equal(model.weight, original)


def test_parameter_filters_leave_excluded_tensors_untouched() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    second = model[1].weight.detach().clone()
    perturber = GaussianPerturber(model, include=r"^0\.")
    with perturber.perturb(seed=1, sigma=0.5):
        assert torch.equal(model[1].weight, second)
