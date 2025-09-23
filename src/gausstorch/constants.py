import os

DOCS_PATH = os.path.join(os.environ["USERPROFILE"], "Documents").replace('\\', '/')

CODE_VERSION = '1.0'
SIM_DATA_DIR_PATH = DOCS_PATH + f'/Simulations_GaussTorch_v{CODE_VERSION}'

SYST_VARS_KEYS_WITHOUT_BIASES = [
    'detuning',
    'eA_real', 'eA_imag',
    'g_real', 'g_imag',
    'gs_real', 'gs_imag',
    'k_int', 'k_ext'
]
SYST_VAR_BIAS_KEYS = [
    'W_0', 'W_bias',
    'theta_bias_real', 'theta_bias_imag',
    'phi_0', 'phi_bias',
]
SYST_VARS_KEYS_WITH_BIASES = SYST_VARS_KEYS_WITHOUT_BIASES + SYST_VAR_BIAS_KEYS