"""
Module constaining `Qsyst` class. An instance contains the drive, detuning, coupling and dissipation parameters of a M mode system.

The `Qsyst` methods will then be used to compute the displacement and covariance matrices of the system after time evolutions under input encoding.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from math import prod, factorial
import matplotlib.pyplot as plt

from gausstorch.utils.operations import (
    torch_block,
    cholesky_inverse_det,
    truncate_alpha,
    truncate_sigma,
)
from gausstorch.utils.display import (
    setup_tex,
    plot_evolution_N,
    plot_evolution_fock,
    fock_states_to_str_list,
)
from gausstorch.utils.bcolors import bcolors
from gausstorch.utils.loop_hafnian_torch import loop_hafnian
from gausstorch.constants import SYST_VARS_KEYS_WITH_BIASES

torch.set_default_dtype(torch.float64)


def init_pars_default(M):
    """

    Args:
        M: Number of modes.

    Returns:
        output: Default parameter values for M modes.

    """
    g_shape = M * (M - 1) // 2
    eA = M * 1e5
    output = {
        "M": M,
        # learnable parameters
        "W_0": torch.ones(1),
        "W_bias": torch.zeros(1),
        "theta_bias_real": torch.zeros(1),
        "theta_bias_imag": torch.zeros(1),
        "phi_0": torch.zeros(1),
        "phi_bias": torch.zeros(1),
        "detuning": torch.zeros(M),
        "eA_real": eA.real * torch.ones(M),
        "eA_imag": eA.imag * torch.ones(M),
        "g_real": 2 * torch.pi * 100e6 * torch.ones(g_shape),
        "g_imag": torch.zeros(g_shape),
        "gs_real": 2 * torch.pi * 20e6 * torch.ones(g_shape),
        "gs_imag": torch.zeros(g_shape),
        "k_int": 0 * torch.ones(M),
        "k_ext": 2 * torch.pi * 2e6 * torch.ones(M),
        # other parameters
        "t_i": torch.tensor(1e-7),
    }
    return output


class Qsyst(nn.Module):
    def __init__(
        self, init_pars: dict, learnable_vars: list = [], init_print: bool = False
    ):
        """

        Args:
            init_pars: Dict containing drive, detuning, coupling, dissipation parameters, and time interval. See the init_pars_default() function for a template.
            learnable_vars: List containing the parameters to learn (with autograd enabled). By default, no parameter has gradient computation enabled.
            init_print: Bool checking whether to print a message during instance creation or not.
        """
        super().__init__()
        self.init_pars = init_pars
        self.learnable_vars = learnable_vars
        if init_print:
            print(f"{bcolors.BOLD}initializing Qsyst model{bcolors.ENDC}")
        # System parameters are store in the OrderedDict syst_vars
        # Hyper-parameters like the evolution time t_i are stored in other_pars.
        # deepcopy init_par to prevent memory issues due to dictionaries.
        self.syst_vars = deepcopy(self.make_syst_vars())
        self.other_pars = deepcopy(self.make_other_pars())

        # set no_grad for non-learnable parameters
        for key, value in self.syst_vars.items():
            if key not in learnable_vars:
                self.syst_vars.update(
                    {key: nn.Parameter(value.data, requires_grad=False)}
                )

        # useful constants to keep
        self.M = self.init_pars["M"]
        self.g_shape = self.M * (self.M - 1) // 2
        self.alpha0 = torch.zeros((2 * self.M, 1), dtype=torch.complex128)
        self.sigma0 = (1 / 2) * torch.eye(2 * self.M, dtype=torch.complex128)

    def make_syst_vars(self):
        """

        Returns:
            syst_vars: nn.ParameterDict with the parameters to learn.

        """
        syst_vars = nn.ParameterDict(
            {
                key: nn.Parameter(self.init_pars[key], requires_grad=True)
                for key in SYST_VARS_KEYS_WITH_BIASES
            }
        )
        return syst_vars

    def make_other_pars(self):
        """

        Returns:
            other_pars: dict with parameters which can never be learned

        """
        other_pars = {key: self.init_pars[key] for key in ["t_i"]}
        return other_pars

    def create_coupling_matrices(
        self,
        detuning: torch.Tensor,
        g_cplx: torch.Tensor,
        gs_cplx: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            detuning (torch.Tensor): 1D tensor containing the detunings of the nearly resonant drives
            g_cplx (torch.Tensor): 1D tensor containing the photon conversion rates
            gs_cplx (torch.Tensor): 1D tensor containing the two-mode squeezing rates

        Returns:
            torch.Tensor: Coupling matrix (defined in my PhD thesis at equation 5.46)

        """
        G = torch.diag(detuning).type(torch.complex128)
        i, j = 0, 1
        for g_ in g_cplx:
            G[i, j] = g_
            G[j, i] = torch.conj(g_)
            j += 1
            if j == self.M:
                i += 1
                j = i + 1

        Gs = torch.zeros((self.M, self.M), dtype=torch.complex128)
        i, j = 0, 1
        for gs_ in gs_cplx:
            Gs[i, j] = gs_ / 2
            Gs[j, i] = gs_ / 2
            j += 1
            if j == self.M:
                i += 1
                j = i + 1
        L0 = -1j * torch_block(G, 2 * Gs, -2 * Gs.conj().t(), -G.t())
        return L0

    def return_theta_xmask(self, theta_key: str) -> tuple:
        """

        Args:
            theta_key (str): Key of encoding parameter theta

        Returns:
            tuple: `theta_0` the syst.syst_vars parameter associated to theta_key, and `xmask_shape` the shape of the encoding parameter, for the later linear encoding with xmask

        """
        if theta_key == "eA":
            theta_0 = self.syst_vars.eA_real + 1j * self.syst_vars.eA_imag
            xmask_shape = self.M
        elif theta_key == "detuning":
            theta_0 = self.syst_vars.detuning
            xmask_shape = self.M
        elif theta_key == "g":
            theta_0 = self.syst_vars.g_real + 1j * self.syst_vars.g_imag
            xmask_shape = self.g_shape
        elif theta_key == "gs":
            theta_0 = self.syst_vars.gs_real + 1j * self.syst_vars.gs_imag
            xmask_shape = self.g_shape
        else:
            raise AssertionError
        return theta_0, xmask_shape

    def encode_theta(
        self,
        theta_0: torch.torch.Tensor,
        x_mask: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        encode_phase: bool,
    ) -> torch.Tensor:
        """

        Args:
            theta_0 (torch.Tensor): Encoding parameter value, before encoding the inputs x_mask
            x_mask (torch.Tensor): Input values, in a 1-D tensor of the same shape as theta_0
            x_min (torch.Tensor): Minimum input value (used to rescale the input between x_min and x_max)
            x_max (torch.Tensor): Maximum input value (used to rescale the input between x_min and x_max)
            encode_phase (bool): Bool deciding whether to encode into the absolute value or the phase of the encoding parameter (equations 5.105 and 5.106 in my thesis)

        Returns:
            torch.Tensor: Encoded parameter value

        """
        theta_bias = (
            self.syst_vars.theta_bias_real + 1j * self.syst_vars.theta_bias_imag
        )
        # normalize x_mask values between 0 and 1
        x_mask = (x_mask.clone() - x_min) / (x_max - x_min)
        if encode_phase:
            # phase encoded between 0 and pi phase
            theta_encoded = theta_bias + torch.mul(
                theta_0.clone(),
                torch.exp(
                    1j * (self.syst_vars.phi_0 * x_mask + self.syst_vars.phi_bias)
                ),
            )  # should have pi in phase
        else:  # encode in the amplitude
            theta_encoded = torch.mul(theta_0.clone(), x_mask) + theta_bias

        return theta_encoded

    def alpha_sigma_evolution_part_1(
        self, theta_key: str, theta_encoded: torch.Tensor
    ) -> tuple:
        """Diagonalize the coupling matrix with dissipations.

        Args:
            theta_key (str): Key of the encoding parameters.
            theta_encoded (torch.Tensor): Value of the encoded parameter.

        Returns:
            tuple: `lambda_F, U, Uinv, K_int, K_ext`, the eigenvalue decomposition and dissipation matrices
        """
        detuning = self.syst_vars.detuning
        g_cplx = self.syst_vars.g_real + 1j * self.syst_vars.g_imag
        gs_cplx = self.syst_vars.gs_real + 1j * self.syst_vars.gs_imag

        # The encoding parameter takes the place of the Qsyst parameter.
        if theta_key == "g":
            g_cplx = theta_encoded
        elif theta_key == "gs":
            gs_cplx = theta_encoded

        L0 = self.create_coupling_matrices(
            detuning=detuning, g_cplx=g_cplx, gs_cplx=gs_cplx
        )

        K_ext = torch.diag(
            torch.cat((self.syst_vars.k_ext, self.syst_vars.k_ext), dim=0)
        ).type(torch.complex128)
        K_int = torch.diag(
            torch.cat((self.syst_vars.k_int, self.syst_vars.k_int), dim=0)
        ).type(torch.complex128)
        K = K_int + K_ext
        F_ = L0 - (K / 2)  # F' in my thesis
        # trick for faster integral computation. Backward only works if eigenvalues are real
        lambda_F, U = torch.linalg.eig(F_)
        Uinv = torch.inverse(U)

        return lambda_F, U, Uinv, K_int, K_ext

    def alpha_sigma_evolution_part_2(
        self,
        theta_key: str,
        theta_encoded: torch.Tensor,
        t: torch.Tensor,
        alpha_i: torch.Tensor,
        sigma_i: torch.Tensor,
        lambda_F: torch.Tensor,
        U: torch.Tensor,
        Uinv: torch.Tensor,
        K_int: torch.Tensor,
        K_ext: torch.Tensor,
    ) -> tuple:
        """
        Computes the displacement (alpha) and covariance matrix (sigma) using the coupling matrix eigenvalue decomposition, and the evolution time t.

        Args:
            theta_key (str): Key of the encoding parameter.
            theta_encoded (torch.Tensor): Encoded parameter value.
            t (torch.Tensor): Duration of the time evolution.
            alpha_i (torch.Tensor): Displacement value.
            sigma_i (torch.Tensor): Covariance matrix.
            lambda_F (torch.Tensor): Coupling matrix eigenvalue
            U (torch.Tensor): Coupling matrix eigenvectors (each column is an eigenvector)
            Uinv (torch.Tensor): Inverse of `U`
            K_int (torch.Tensor): Diagonal matrix containing internal dissipations
            K_ext (torch.Tensor): Diagonal matrix containing external dissipations

        Returns:
            tuple: Displacement vector and Covariance matrix

        """
        M = self.M
        eA_cplx = self.syst_vars.eA_real + 1j * self.syst_vars.eA_imag
        if theta_key == "eA":
            eA_cplx = theta_encoded
        eA_cplx = eA_cplx.view(M, 1)
        # minus sign to be the same as cascaded formalism. Otherwise, can put plus sign. All in all not important
        A_in = -torch.vstack((eA_cplx, eA_cplx.conj())).to(torch.complex128)
        sigma0 = (1 / 2) * torch.eye(2 * M, dtype=torch.complex128)
        sigma_i = sigma_i.type(torch.complex128)
        K = K_int + K_ext

        # Now calculation involving t
        F_t = (
            U @ torch.diag(torch.exp(lambda_F * t)) @ Uinv
        )  # F_t = torch.matrix_exp(F_ * t)

        I1 = torch.diag((1 / lambda_F) * (-1 + torch.exp(lambda_F * t)))
        alpha_output = F_t @ alpha_i + U @ I1 @ Uinv @ torch.sqrt(K_ext) @ A_in

        # I2 is the integral over [0,t] of exp((L-kappa/2)*tau) @ exp((L-kappa/2)*tau).T
        #
        # non vectorized computation of I2, but easier to understand:
        P = Uinv @ K @ Uinv.conj().t()
        I2 = torch.zeros((2 * M, 2 * M), dtype=torch.complex128)
        for i in range(2 * M):
            for j in range(2 * M):
                lambda_F_sum = lambda_F[i] + lambda_F[j].conj()
                I2[i, j] = P[i, j] * (torch.exp(lambda_F_sum * t) - 1) / lambda_F_sum
        #
        # vectorized computation of I2:
        # def sum_conj(a, b):
        #     def sum_b(a0):
        #         return (a0 * torch.ones_like(b)) + b.conj()
        #
        #     batched_sum_b = torch.func.vmap(sum_b)
        #     return batched_sum_b(a)
        #
        # lambda_F_sum = sum_conj(lambda_F, lambda_F)
        # P = Uinv @ K @ Uinv.conj().t()
        # I2 = P * (torch.exp(lambda_F_sum * t) - 1) / lambda_F_sum

        sigma_output = (
            F_t @ sigma_i @ F_t.conj().t() + sigma0 @ U @ I2 @ U.conj().t()
        )  # !!! covariance matrix can have complex values
        return alpha_output, sigma_output

    def alpha_sigma_evolution(
        self,
        t: torch.Tensor,
        alpha_i: torch.Tensor,
        sigma_i: torch.Tensor,
        theta_key: str,
        theta_encoded: torch.Tensor,
    ) -> tuple:
        """Computes a gaussian state displacement and covariance matrix after a time evolution

        Args:
            t: duration of evolution from gaussian state (d_i, sigma_i) state to measurement
            alpha_i: initial displacement
            sigma_i: initial covariance
            theta_key: key of the encoding parameter
            theta_encoded: value of the encoding parameter
        Returns:
            tuple: new gaussian state (alpha_output, sigma_output)
        Args:
            t (torch.Tensor): Duration of evolution from gaussian state (d_i, sigma_i) state to measurement.
            alpha_i (torch.Tensor): Initial displacement vector.
            sigma_i (torch.Tensor): Initial covariance matrix.
            theta_key (str): Name of the encoding parameter
            theta_encoded (torch.Tensor): Value of the encoded parameter.

        Returns:
            tuple: New gaussian state `(alpha_output, sigma_output)`.
        """

        # preliminary matrix computations
        lambda_F, U, Uinv, K_int, K_ext = self.alpha_sigma_evolution_part_1(
            theta_key=theta_key, theta_encoded=theta_encoded
        )
        alpha_output, sigma_output = self.alpha_sigma_evolution_part_2(
            theta_key=theta_key,
            theta_encoded=theta_encoded,
            t=t,
            alpha_i=alpha_i,
            sigma_i=sigma_i,
            lambda_F=lambda_F,
            U=U,
            Uinv=Uinv,
            K_int=K_int,
            K_ext=K_ext,
        )
        return alpha_output, sigma_output

    @staticmethod
    def prob_with_shots(prob: torch.Tensor, n_shots: int) -> torch.Tensor:
        """Computes a probability for a given number of measurement shots with the binomial law

        Args:
            prob (torch.Tensor): Probability of a Gaussian Boson Sampling (GBS) probability
            n_shots (int): Number of measurement shots

        Returns:
            torch.Tensor: GBS probability with shot measurement noise
        """
        new_prob = prob + torch.randn(1) * torch.sqrt(prob * (1 - prob) / n_shots)
        return new_prob

    def prob_gbs(
        self, alpha: torch.Tensor, sigma: torch.Tensor, n: list, n_shots: int = None
    ) -> torch.Tensor:
        """This function uses the global GBS formula to calculate P(n) from the 1st and 2nd moments.

        Args:
            alpha (torch.Tensor): Field operator displacement vector
            sigma (torch.Tensor): Field operator covariance matrix
            n (list): List containing the number of photons to measure in each mode
            n_shots (int, optional): Number of measurement shots. Defaults to None, in which case perfectly accurate estimations are considered.

        Returns:
            torch.Tensor: Fock state occupation probability
        """
        nn2 = n + n
        M = self.M
        id1 = torch.eye(M)
        id2 = torch.eye(2 * M)
        # block torch tensor
        X = torch_block(torch.zeros(M, M), id1, id1, torch.zeros(M, M)).type(
            torch.complex128
        )
        sigmaQ = sigma + 0.5 * id2
        sigmaQ_inv, sigmaQ_det = cholesky_inverse_det(sigmaQ)
        O = (id2 - sigmaQ_inv).type(torch.complex128)
        A = X @ O
        sigmaQ_inv = sigmaQ_inv.type(torch.complex128)
        gamma = (alpha.conj().t() @ sigmaQ_inv).squeeze()
        lhaf_A = loop_hafnian(A, D=gamma, reps=nn2)
        # print(f'dtype of lhaf_A: {lhaf_A.dtype}')
        result = (
            lhaf_A
            * torch.exp(-0.5 * alpha.conj().T @ sigmaQ_inv @ alpha)
            / (torch.sqrt(sigmaQ_det) * prod([factorial(ni) for ni in n]))
        )
        # print(f'dtype of result: {result.squeeze().real.dtype}')
        result = result.real
        if n_shots is not None:
            result = self.prob_with_shots(result, n_shots)
        return result.squeeze()

    def prob_gbs_partial_trace(
        self,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        n: list,
        modes_kept: list,
        n_shots: int = None,
    ):
        """This function uses the GBS formula to calculate P(n) from the 1st and 2nd moments.
        All the modes except the ones contained in modes_kept are traced.
        n is the list containing the photon combination to measure, after the partial trace.

        Example:
            `M = 3`
            `osc_i = (0, 2)`
            `n = (2, 4)`
        This means you trace out mode 1, and compute the probability of
        measuring 2 photons in mode 0, and 4 photons in mode 2.

        Args:
            alpha (torch.Tensor): Field operator displacement vector.
            sigma (torch.Tensor): Field operator covariance matrix.
            n (list): List containing the number of photons to measure in each considered mode
            modes_kept (list): List of considered modes
            n_shots (int, optional): Number of measurement shots. Defaults to None, in which case perfectly accurate estimations are considered.

        Returns:
            _type_: _description_
        """

        sigma_new = truncate_sigma(sigma, modes_kept)
        alpha_new = truncate_alpha(alpha, modes_kept)

        nn2 = n + n
        M = len(modes_kept)
        id1 = torch.eye(M)
        id2 = torch.eye(2 * M)
        # block torch tensor
        T = torch_block(torch.zeros(M, M), id1, id1, torch.zeros(M, M)).type(
            torch.complex128
        )
        sigmaQ = sigma_new + 0.5 * id2
        sigmaQ_inv, sigmaQ_det = cholesky_inverse_det(sigmaQ)
        O = (id2 - sigmaQ_inv).type(torch.complex128)
        A = T @ O
        sigmaQ_inv = sigmaQ_inv.type(torch.complex128)
        gamma = (alpha_new.conj().t() @ sigmaQ_inv).squeeze()
        lhaf_A = loop_hafnian(A, D=gamma, reps=nn2)
        result = (
            lhaf_A
            * torch.exp(-0.5 * alpha_new.conj().T @ sigmaQ_inv @ alpha_new)
            / (torch.sqrt(sigmaQ_det) * prod([factorial(ni) for ni in n]))
        )
        if n_shots is not None:
            result = self.prob_with_shots(result, n_shots)
        return result.squeeze().real

    def evolution_N(
        self,
        theta_key: str = "eA",
        x_val: torch.Tensor = torch.tensor(1),
        x_min: torch.Tensor = torch.tensor(0),
        x_max: torch.Tensor = torch.tensor(1),
        encode_phase: bool = False,
        res: int = 1_000,
        compute_plot: bool = True,
        yscale: str = "linear",
        show_plot: bool = True,
        inference_mode: bool = True,
        return_vals: bool = False,
        return_tspan: bool = False,
    ):
        """Plots or computes the time evolution of the system during a time interval

        Args:
            x_val (torch.Tensor, optional): Input value. Defaults to torch.tensor(1).
            x_min (torch.Tensor, optional): Minimum input value (for input rescaling). Defaults to torch.tensor(0).
            x_max (torch.Tensor, optional): Maximum input value (for input rescaling). Defaults to torch.tensor(1).
            encode_phase (bool, optional): If `True`, the input is encoded into the phase of the encoding parameter. Defaults to False.
            res (int, optional): Number of points to compute. Defaults to 1_000.
            compute_plot (bool, optional): If `True`, the plot is computed. Defaults to True.
            yscale (str, optional): Choice of yscale. Defaults to 'linear'.
            show_plot (bool, optional): If `True`, plot is shown with `plt.show()`. Defaults to True.
            inference_mode (bool, optional): If `True`, no gradients are computed. Defaults to True.
            return_vals (bool, optional): If `True`, return photon number values. Defaults to False.
            return_tspan (bool, optional): If `True`, also return discrete time values. Defaults to False.

        Returns:
            _type_: Either None, or photon number values with discrete time values

        """
        torch.set_num_threads(1)
        torch.inference_mode(inference_mode)

        # encode the input x into theta_0
        theta_0, x_mask_shape = self.return_theta_xmask(theta_key)
        x_mask = x_val * torch.ones(x_mask_shape, dtype=torch.complex128)
        theta_encoded = self.encode_theta(
            theta_0=theta_0,
            x_mask=x_mask,
            x_min=x_min,
            x_max=x_max,
            encode_phase=encode_phase,
        )
        tspan = torch.linspace(0, self.other_pars["t_i"], res)
        means = torch.zeros((res, self.M))
        # perform the eigenvalue decomposition only once, then use to compute alpha and sigma at all times t from tspan.
        lambda_F, U, Uinv, K_int, K_ext = self.alpha_sigma_evolution_part_1(
            theta_key=theta_key, theta_encoded=theta_encoded
        )
        for i, t in enumerate(tspan):
            alpha_t, sigma_t = self.alpha_sigma_evolution_part_2(
                theta_key=theta_key,
                theta_encoded=theta_encoded,
                t=t,
                alpha_i=self.alpha0,
                sigma_i=self.sigma0,
                lambda_F=lambda_F,
                U=U,
                Uinv=Uinv,
                K_int=K_int,
                K_ext=K_ext,
            )

            for j in range(self.M):
                means[i, j] = (
                    torch.real(sigma_t[j, j]) + (torch.abs(alpha_t[j]) ** 2) - 0.5
                )
        # plot means
        tspan_renormalized = tspan * torch.mean(
            self.syst_vars.k_ext
        )  # renorm by kappa average
        if compute_plot:
            tspan_np = tspan_renormalized.detach().numpy()
            means_np = means.detach().numpy()
            fig, ax = plot_evolution_N(
                tspan_np,
                means_np,
                width_ratio=0.48,
                xlabel=r"time $\times \kappa$",
                yscale=yscale,
            )
            if show_plot:
                # print(f"at time {tspan[-1]},\n"
                #       f"final alpha_t: \n{alpha_t}\n"
                #       f"final sigma_t: \n{sigma_t}")
                plt.show()
        torch.inference_mode(False)
        if return_vals:
            if return_tspan:
                return tspan, means
            else:
                return means

    def evolution_fock(
        self,
        fock_combs_per_mode_comb: dict = None,
        theta_key: str = "eA",
        x_val: torch.Tensor = torch.tensor(1),  # input, of value between 0 and 1
        x_min: torch.Tensor = torch.tensor(0),
        x_max: torch.Tensor = torch.tensor(1),
        encode_phase: bool = False,
        res: int = 1_000,
        compute_plot: bool = True,
        show_plot: bool = True,
        inference_mode: bool = True,
        return_vals: bool = False,
    ):
        """Plots or compute the Fock state occupation probabilities during a time interval

        Args:
            fock_combs_per_mode_comb (dict, optional): Fock state combinations to plot. Defaults to None.
            theta_key (str, optional): Name of the encoding parameter. Defaults to 'eA'.
            x_val (torch.Tensor, optional): Input value. Defaults to torch.tensor(1).
            x_min (torch.Tensor, optional): Minimum input value (for input rescaling). Defaults to torch.tensor(0).
            x_max (torch.Tensor, optional): Maximum input value (for input rescaling). Defaults to torch.tensor(1).
            encode_phase (bool, optional): If `True`, input is encoded into the phase of the encoding parameter. Defaults to False.
            res (int, optional): Number of discrete time steps. Defaults to 1_000.
            compute_plot (bool, optional): If `True`, compute the plot. Defaults to True.
            show_plot (bool, optional): If `True`, plot is shown with plt.show(). Defaults to True.
            inference_mode (bool, optional): If `True`, gradient calculations are disabled. Defaults to True.
            return_vals (bool, optional): If `True`, return Fock states values. Defaults to False.

        Returns:
            _type_: Either None, or torch.Tensor containing Fock state values
        """
        torch.set_num_threads(1)

        torch.inference_mode(inference_mode)
        num_probs = sum(
            [
                len(fock_combinations)
                for fock_combinations in fock_combs_per_mode_comb.values()
            ]
        )

        # encode the input x into theta_0
        theta_0, x_mask_shape = self.return_theta_xmask(theta_key)
        x_mask = x_val * torch.ones(x_mask_shape, dtype=torch.complex128)
        theta_encoded = self.encode_theta(
            theta_0=theta_0,
            x_mask=x_mask,
            x_min=x_min,
            x_max=x_max,
            encode_phase=encode_phase,
        )
        tspan = torch.linspace(0, self.other_pars["t_i"], res)
        probs = torch.zeros((res, num_probs))
        # perform the eigenvalue decomposition only once, then use to compute alpha and sigma at all times t from tspan.
        lambda_F, U, Uinv, K_int, K_ext = self.alpha_sigma_evolution_part_1(
            theta_key=theta_key, theta_encoded=theta_encoded
        )
        for i, t in enumerate(tspan):
            alpha_t, sigma_t = self.alpha_sigma_evolution_part_2(
                theta_key=theta_key,
                theta_encoded=theta_encoded,
                t=t,
                alpha_i=self.alpha0,
                sigma_i=self.sigma0,
                lambda_F=lambda_F,
                U=U,
                Uinv=Uinv,
                K_int=K_int,
                K_ext=K_ext,
            )
            prob_counter = 0
            for mode_comb, fock_combinations in fock_combs_per_mode_comb.items():
                for fock_combination in fock_combinations:
                    p = self.prob_gbs_partial_trace(
                        alpha_t, sigma_t, fock_combination, mode_comb
                    )
                    probs[i, prob_counter] = p
                    prob_counter += 1
        torch.inference_mode(False)
        # plot probs
        if compute_plot:
            prob_means = torch.mean(probs, dim=0).tolist()
            tspan_renormalized = tspan * torch.mean(
                self.syst_vars.k_ext
            )  # renorm by kappa average
            tspan_np = tspan_renormalized.detach().numpy()
            probs_np = probs.detach().numpy()
            labels = fock_states_to_str_list(fock_combs_per_mode_comb)
            # labels = [prob_notation + rf"\n avg$={avg:.4e}$" for prob_notation, avg in zip(labels, prob_means)]
            labels = [prob_notation for prob_notation, avg in zip(labels, prob_means)]
            if len(labels) == 1:
                labels = labels[0]
            fig, ax = plot_evolution_fock(
                tspan=tspan_np,
                probs=probs_np,
                labels=labels,
                width_ratio=0.48,
                xlabel=r"time $\times \kappa$",
                ylabel="Probability",
            )
            if show_plot:
                plt.show()
        if return_vals:
            return probs
