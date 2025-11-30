"""
Test module for the :py:mod:`gausstorch.utils.operations` module.
"""

import torch
from gausstorch.utils.operations import torch_block, wigner, wigner_2d_map

torch.set_default_dtype(torch.float64)


def test_torch_block(n=3):
    """Check the :py:func:`gausstorch.utils.operations.torch_block` function works properly

    Args:
        n (int, optional): Size of component tensors. Defaults to 3.
    """
    A, B, C, D = (
        torch.zeros((n, n)),
        torch.zeros((n, n)),
        torch.zeros((n, n)),
        torch.zeros((n, n)),
    )
    assert torch_block(A, B, C, D).equal(torch.zeros(2 * n, 2 * n))


def test_wigner(
    d: torch.Tensor = torch.zeros((2, 1)),
    alpha: torch.Tensor = torch.zeros((2, 1)),
    sigma: torch.Tensor = 0.5 * torch.eye(2),
):
    """Checks if :py:func:`gausstorch.utils.operations.wigner` evaluation at zero of vacuum state is equal to 1/pi.

    Args:
        d (torch.Tensor): Phase space location. Defaults to torch.zeros((2, 1)).
        alpha (torch.Tensor): Quadrature displacement vector. Defaults to torch.zeros((2, 1)).
        sigma (torch.Tensor): Quadrature covariance matrix. Defaults to 0.5*torch.eye(2).
    """

    assert wigner(d, alpha, sigma).isclose(torch.tensor(1 / torch.pi))
