import torch

from gausstorch.utils.bcolors import bcolors


def torch_block(A, B, C, D):
    """
    A, B, C, D: torch.tensor(N, N)
    create block torch.tensor(dtype=torch.complex64)
    [[A, B],
    [C, D]]
    """
    x, y = torch.cat((A, B, C, D), dim=1).t().chunk(2)
    return torch.cat((x, y), dim=1).t()


def cholesky_inverse_det(M):
    """
    faster and more accurate than torch.det and torch.inverse for symmetric positive definite matrices
    """
    # global num_cholesky_computations
    # global M_eigvals_list
    try:
        L = torch.linalg.cholesky(M, upper=False)
        M_det = torch.prod(torch.diag(L)) ** 2
        return torch.cholesky_inverse(L, upper=False), M_det
    except torch._C._LinAlgError:
        raise torch._C._LinAlgError(f"{bcolors.FAIL}caught torch cholesky linalgerror\n"
                                    f"M value giving this error:\n{M}\n"
                                    f"M eigenvalues: \n{torch.linalg.eigvals(M)}{bcolors.ENDC}")
        # return torch.inverse(M), torch.det(M)


# region Folded manipulation of gaussian moments

def moments_to_quad_moments(alpha, sigma):
    """
    transform ani&crea moments to quadrature moments
    :param alpha: 2M * 1 tensor
    :param sigma: 2M*2M tensor
    :return: alpha and sigma of quadratures
    """
    M = alpha.shape[0] // 2
    gamma_dag = (1 / np.sqrt(2)) * torch_block(
        torch.eye(M), torch.eye(M), -1j * torch.eye(M), 1j * torch.eye(M)
    )  # also inverse of gamma
    gamma = gamma_dag.t().conj()
    alpha_r = gamma_dag @ alpha
    sigma_r = gamma_dag @ sigma @ gamma
    return alpha_r, sigma_r

def truncate_sigma(sigma, modes_kept, M):
    """
    sigma: full covariance matrix
    modes_kept: modes to keep, truncate the other ones
    M: total number of modes
    """
    rows = modes_kept  # 1d iterable of rows to keep
    columns = modes_kept  # 1d iterable of columns to keep
    M_kept = len(modes_kept)
    sigma_truncated = torch.empty((2 * len(rows), 2 * len(columns)), dtype=torch.complex128)
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


def truncate_disp(disp, modes_kept, M):
    """
    disp: full displacement vector
    modes_kept: modes to keep, truncate the other ones
    M: total number of modes
    """
    rows = modes_kept  # 1d list of rows to keep
    M_kept = len(modes_kept)
    disp_truncated = torch.empty((2 * len(rows), 1), dtype=torch.complex128)
    for i, row in enumerate(rows):
        disp_truncated[i, 0] = torch.select(
            torch.select(disp, dim=0, index=row), dim=0, index=0
        )
        disp_truncated[i + M_kept, 0] = torch.select(
            torch.select(disp, dim=0, index=row + M), dim=0, index=0
        )
    return disp_truncated

# endregion Folded manipulation of gaussian moments

# region Folded Wigner function

def wigner(d, alpha, sigma):
    """
    Use quadrature 1st and 2nd moments, not creation and annihilation operators
    :param d: 2M * 1 tensor
    :param alpha: 2M * 1 tensor
    :param sigma: 2M*2M tensor
    :return: wigner function at d coordinate
    """
    M = alpha.shape[0] // 2
    return torch.exp(-0.5 * (alpha - d).t() @ torch.linalg.inv(sigma) @ (alpha - d)) / (
            ((2 * torch.pi) ** M) * torch.sqrt(torch.linalg.det(sigma)))


def wigner_2d_map(alpha, sigma, xvec: torch.linspace(-5, 5, 50), yvec: torch.linspace(-5, 5, 50)):
    assert alpha.shape == torch.Size([2, 1]) and sigma.shape == torch.Size([2, 2])
    assert len(xvec.shape) == 1 and len(yvec.shape) == 1
    W = torch.empty((xvec.shape[0], yvec.shape[0]))
    for i_x, x in enumerate(xvec):
        for i_y, y in enumerate(yvec):
            d = torch.tensor([[x], [y]])
            w = wigner(d, alpha, sigma)
            W[i_x, i_y] = w.real
    return W

# endregion Folded Wigner function
