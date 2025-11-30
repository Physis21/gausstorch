"""
This module contains useful functions which operate on the `torch.Tensor` objects manipulated in the `gausstorch` package.
"""

import torch
import numpy as np

from gausstorch.utils.bcolors import bcolors

torch.set_default_dtype(torch.float64)


def torch_block(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
) -> torch.Tensor:
    """Creates a block tensor with 4 square tensors of equal sizes

    Args:
        A (torch.Tensor): Square tensor
        B (torch.Tensor): Square tensor
        C (torch.Tensor): Square tensor
        D (torch.Tensor): Square tensor

    Returns:
        torch.Tensor: Block tensor [[A, B], [C, D]]
    """

    x, y = torch.cat((A, B, C, D), dim=1).t().chunk(2)
    return torch.cat((x, y), dim=1).t()


def cholesky_inverse_det(M: torch.Tensor) -> tuple:
    """This function is faster and more accurate than torch.det and torch.inverse for symmetric positive definite matrices

    Args:
        M (torch.Tensor): _description_

    Raises:
        torch._C._LinAlgError: _description_

    Returns:
        tuple: Inverse of M, and determinant of M
    """
    # global num_cholesky_computations
    # global M_eigvals_list
    try:
        L = torch.linalg.cholesky(M, upper=False)
        M_det = torch.prod(torch.diag(L)) ** 2
        return torch.cholesky_inverse(L, upper=False), M_det
    except torch._C._LinAlgError:
        raise torch._C._LinAlgError(
            f"{bcolors.FAIL}caught torch cholesky linalgerror\n"
            f"M value giving this error:\n{M}\n"
            f"M eigenvalues: \n{torch.linalg.eigvals(M)}{bcolors.ENDC}"
        )
        # return torch.inverse(M), torch.det(M)


# region Folded Matrix slicing


def slice_2d_nn(M: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Used in slice_2d_sigma to get the value of the (n,n) shape tensor M at indexes i and j

    Args:
        M (torch.Tensor): (n,n) shape tensor
        i (int): First index value
        j (int): Second index value

    Returns:
        torch.Tensor: Value of M[i,j]
    """
    # use in slice_2d_sigma
    return torch.select(torch.select(M, 1, i), 0, j)


def slice_2d_n1(M: torch.Tensor, i: int) -> torch.Tensor:
    """Used in slice_2d_alpha to get the value of the (n,1) shape tensor M at index i

    Args:
        M (torch.Tensor): (n,1) shape tensor
        i (int): Index value

    Returns:
        torch.Tensor: value of M[i, 0]
    """
    # use in slice_2d_alpha
    return torch.select(M, 0, i)


def slice_2d_sigma(M: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Use to trace the covariance matrix over modes not i and j. Will usually be called with j = i + (M.shape[0] // 2).

    Args:
        M (torch.Tensor): A square tensor. In the context of this function's use, M is a covariance matrix
        i (int): first index to trace lines and columns over
        j (int): second index to trace lines and columns over

    Returns:
        torch.Tensor: [[M[i,i], M[i,j]], [M[j,i], M[j,j]]]
    """
    M00 = slice_2d_nn(M, i, i).view((1, 1))
    M01 = slice_2d_nn(M, i, j).view((1, 1))
    M10 = slice_2d_nn(M, j, i).view((1, 1))
    M11 = slice_2d_nn(M, j, j).view((1, 1))

    M_line0 = torch.cat((M00, M01), dim=1)
    M_line1 = torch.cat((M10, M11), dim=1)
    M_traced = torch.cat((M_line0, M_line1), dim=0)
    return M_traced


def slice_2d_alpha(M: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Use to trace the displacement vector over modes not i and j.

    Args:
        M (torch.Tensor): A (n,1) shape tensor. In the context of this function's use, M is a displacement vector
        i (int): first index to trace lines over
        j (int): second index to trace lines over

    Returns:
        torch.Tensor: [[M[i,0]], [M[j]]]
    """
    #
    M00 = slice_2d_n1(M, i).view((1, 1))
    M10 = slice_2d_n1(M, j).view((1, 1))

    M_traced = torch.cat((M00, M10), dim=0)
    return M_traced


# endregion Folded Matrix slicing


# region Folded manipulation of gaussian moments


def moments_to_quad_moments(alpha: torch.Tensor, sigma: torch.Tensor) -> tuple:
    """Transforms field operator moments to quadrature moments

    :param alpha:
    :param sigma: 2M*2M tensor
    :return: alpha and sigma of quadratures
    Args:
        alpha (torch.Tensor): (2M,1)-shape tensor, representing the displacement vector
        sigma (torch.Tensor): (2M,2M)-shape tensor, representing the covariance matrix

    Returns:
        tuple: quadrature displacement and covariance matrix
    """

    M = alpha.shape[0] // 2
    gamma_dag = (1 / np.sqrt(2)) * torch_block(
        torch.eye(M), torch.eye(M), -1j * torch.eye(M), 1j * torch.eye(M)
    )  # also inverse of gamma
    gamma = gamma_dag.t().conj()
    alpha_r = gamma_dag @ alpha
    sigma_r = gamma_dag @ sigma @ gamma
    return alpha_r, sigma_r


def truncate_sigma(sigma: torch.Tensor, modes_kept: list) -> torch.Tensor:
    """Truncates the covariance matrix, keeping only the modes in `modes_kept`

    Args:
        sigma (torch.Tensor): Full covariance matrix (square tensor)
        modes_kept (list): List of modes to keep

    Returns:
        torch.Tensor: Truncated covariance matrix
    """
    M = sigma.shape[1] // 2
    rows = modes_kept  # 1d iterable of rows to keep
    columns = modes_kept  # 1d iterable of columns to keep
    M_kept = len(modes_kept)
    sigma_truncated = torch.empty(
        (2 * len(rows), 2 * len(columns)), dtype=torch.complex128
    )
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            sigma_truncated[i, j] = torch.select(
                torch.select(sigma, dim=0, index=row), dim=0, index=column
            )
            sigma_truncated[i + M_kept, j] = torch.select(
                torch.select(sigma, dim=0, index=row + M), dim=0, index=column
            )
            sigma_truncated[i, j + M_kept] = torch.select(
                torch.select(sigma, dim=0, index=row), dim=0, index=column + M
            )
            sigma_truncated[i + M_kept, j + M_kept] = torch.select(
                torch.select(sigma, dim=0, index=row + M), dim=0, index=column + M
            )
    return sigma_truncated


def truncate_alpha(alpha: torch.Tensor, modes_kept: list) -> torch.Tensor:
    """Truncates the displacement vector, keeping only the modes in `modes_kept`

    Args:
        alpha (torch.Tensor): Full displacement vector
        modes_kept (list): List of modes to keep

    Returns:
        torch.Tensor: Truncated displacement vector
    """
    M = alpha.shape[0] // 2
    rows = modes_kept  # 1d list of rows to keep
    M_kept = len(modes_kept)
    alpha_truncated = torch.empty((2 * len(rows), 1), dtype=torch.complex128)
    for i, row in enumerate(rows):
        alpha_truncated[i, 0] = torch.select(
            torch.select(alpha, dim=0, index=row), dim=0, index=0
        )
        alpha_truncated[i + M_kept, 0] = torch.select(
            torch.select(alpha, dim=0, index=row + M), dim=0, index=0
        )
    return alpha_truncated


def quad_key_to_index(key: str, M: int) -> int:
    """Returns the index at which to find the quadrature in the displacement or covariance matrix

    Returns:

    Args:
        key (str): Quadrature key (e.g: 'P2')
        M (int): Number of modes

    Raises:
        AssertionError: Checks if the quadrature is X or P

    Returns:
        int: Index at which to find the quadrature in the displacement or covariance matrix
    """
    mode = int(key[1])
    if key[0] == "X":
        index = mode
    elif key[0] == "P":
        index = mode + M
    else:
        raise AssertionError
    return index


def keep_quads_in_alpha_r(
    alpha_r: torch.Tensor, quad_key_1: str, quad_key_2: str
) -> torch.Tensor:
    """Truncates a quadrature displacement vector by keeping only the `quad_key_1` and `quad_key_2` components

    Args:
        alpha_r (torch.Tensor): Quadrature displacement vector
        quad_key_1 (str): Quadrature key (e.g: 'P2')
        quad_key_2 (str): Quadrature key (e.g: 'X0')

    Returns:
        torch.Tensor: Truncated quadrature vector
    """
    M = alpha_r.shape[0] // 2
    ind_1 = quad_key_to_index(quad_key_1, M)
    ind_2 = quad_key_to_index(quad_key_2, M)
    return slice_2d_alpha(alpha_r, ind_1, ind_2)


def keep_quads_in_sigma_r(
    sigma_r: torch.Tensor, quad_key_1: str, quad_key_2: str
) -> torch.Tensor:
    """Truncates a quadrature covariance matrix by keeping only the `quad_key_1` and `quad_key_2` components

    Args:
        sigma_r (torch.Tensor): Quadrature covariance matrix
        quad_key_1 (str): Quadrature key (e.g: 'P2')
        quad_key_2 (str): Quadrature key (e.g: 'X0')

    Returns:
        torch.Tensor: Truncated covariance matrix
    """
    M = sigma_r.shape[0] // 2
    ind_1 = quad_key_to_index(quad_key_1, M)
    ind_2 = quad_key_to_index(quad_key_2, M)
    return slice_2d_sigma(sigma_r, ind_1, ind_2)


# endregion Folded manipulation of gaussian moments

# region Folded Wigner function


def wigner(d: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.tensor:
    """Returns the wigner function at a `d` phase space location

    Args:
        d (torch.Tensor): (2M * 1 tensor) Phase space location
        alpha (torch.Tensor): (2M * 1 tensor) Quadrature displacement
        sigma (torch.Tensor): (2M*2M tensor) Quadrature covariance matrix

    Returns:
        torch.tensor: wigner function at d coordinate
    """
    M = alpha.shape[0] // 2
    return torch.exp(-0.5 * (alpha - d).t() @ torch.linalg.inv(sigma) @ (alpha - d)) / (
        ((2 * torch.pi) ** M) * torch.sqrt(torch.linalg.det(sigma))
    )


def wigner_2d_map(
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    xvec : torch.Tensor = None,
    yvec : torch.Tensor = None,
) -> torch.Tensor:
    """Returns the Wigner quasi-probability distribution

    Args:
        alpha (torch.Tensor): Quadrature displacement vector
        sigma (torch.Tensor): Covariance displacement vector
        xvec (torch.Tensor, optional): 1st quadrature values. Defaults to None.
        yvec (torch.Tensor, optional): 2nd quadrature values. Defaults to None.

    Returns:
        torch.Tensor: 2D Tensor with wigner function values
    """
    if xvec is None:
        xvec = torch.linspace(-5, 5, 50)
    if yvec is None:
        yvec = torch.linspace(-5, 5, 50)
    # assert alpha.shape == torch.Size([2, 1]) and sigma.shape == torch.Size([2, 2])
    # assert len(xvec.shape) == 1 and len(yvec.shape) == 1
    W = torch.empty((xvec.shape[0], yvec.shape[0]), dtype=torch.float64)
    for i_x, x in enumerate(xvec):
        for i_y, y in enumerate(yvec):
            d = torch.tensor([[x], [y]])
            w = wigner(d, alpha, sigma)
            W[i_x, i_y] = w.real
    return W


# endregion Folded Wigner function
