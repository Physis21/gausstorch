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


# region Folded matrix slicing and tracing

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


def truncate_disp(disp, kept_mode_ind, M):
    """
    disp: full displacement vector
    modes_kept: modes to keep, truncate the other ones
    M: total number of modes
    """
    rows = kept_mode_ind  # 1d list of rows to keep
    M_kept = len(kept_mode_ind)
    disp_truncated = torch.empty((2 * len(rows), 1), dtype=torch.complex128)
    for i, row in enumerate(rows):
        disp_truncated[i, 0] = torch.select(
            torch.select(disp, dim=0, index=row), dim=0, index=0
        )
        disp_truncated[i + M_kept, 0] = torch.select(
            torch.select(disp, dim=0, index=row + M), dim=0, index=0
        )
    return disp_truncated

# endregion Folded matrix slicing and tracing
