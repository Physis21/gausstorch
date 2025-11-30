# Source repository for 'gausstorch', a package allowing the simulation of coupled bosonic modes in the Gaussian regime, with PyTorch

- The package documentation is published in [this link](https://gausstorch.readthedocs.io/en/latest/index.html)

## About this repository

- This repository is an overhauled (properly documented and with better coding practices) version of my PhD code, used to **simulate the dynamics of coupled bosonic modes in PyTorch**
- My PhD thesis is available on HAL. [Here is the link](https://theses.hal.science/tel-05383369)
- An instance of the `Qsyst` class in `src/gausstorch/libs/qsyst.py` contains the drive, detuning and coupling parameters between a certain number $M$ of coupled Gaussian modes
  - It will be used to model the dynamics of such a system using the `alpha_sigma_evolution()` methods, and compute the Fock state occupation probabilities (with Gaussian boson sampling) using the `prob_gbs()` method
- During my PhD I used this formalism to train the physical parameters in order to solve _classical_ machine learning tasks
  - Due to time constraints, I have not added the modules allowing for solving of classical machine learning tasks, but rather only the PyTorch package allowing to train the physical parameters
  - If you wish to see how I built learning models on top of the `Qsyst` class, you can download the Zenodo repository for my article [_"Training the parametric interactions in an analog bosonic quantum neural network with Fock basis measurement"_](https://zenodo.org/records/15856611)
- This is an overhauled version with only the minimum sufficient classes and functions, **meant for future PhD students to build up on**
- The documentation has been automatically generated with [sphinx](https://www.sphinx-doc.org/en/master), and is published on [ReadTheDocs](https://docs.readthedocs.com/platform/stable/index.html).

## Requirements & installing

- Have Python **version 3.12.1** installed on your computer. [Download Link](https://www.python.org/downloads/release/python-3121/).
- First clone the Github code into a local directory
- From the command line, move into the root folder of the created directory

```bash
cd root_folder
```

- create a virtual environment from Pycharm or using the command (!!! use the correct Python 3.12.1 version)

```bash
python -m venv .venv
```

- Activate the virtual environment

```bash
source .\venv/bin/activate # on Unix/macOS
.\.venv\Scripts\activate  # on Windows
```

- from the virtual environment run the command:

```bash
pip install -r requirements.txt  # install pip packages used in src/
pip install --editable .  # install the src package
```

- The 'gausstorch' package is now usable from the package `gausstorch`

## READ THE TUTORIAL

Go to [tutorials/qsyst.ipynb](tutorials/qsyst.ipynb) to see how the `Qsyst` class is used to model coupled bosonic modes.

All of the calculations performed in `Qsyst` are detailed in my thesis. I put the relevant sections in the Git repository. [Here is a direct link](<thesis section5-3 + Hamiltonian + Langevin.pdf>).

## Structure of src/gausstorch

- `./src/gausstorch/libs/qsyst`: contains the `Qsyst` class. It contains methods to simulate an arbitrary number of coupled bosonic modes.
  - On initialization, an instance of `Qsyst` requires an `init_pars` argument, which specifies the drive, coupling and dissipation parameters within the modes.
  - Learning models using Gaussian dynamics should be child classes of `Qsyst`
- `./src/gausstorch/utils`: very short and specific functions used throughout functions in `./src/gausstorch/libs`
  - `./src/gausstorch/utils/_loop_hafnian_subroutines.py` and `loop_hafnian.py` are used to compute **loop hafnians**. The functions are adapted to PyTorch from the github repository <https://github.com/jakeffbulmer/gbs> , made by _Jacob F.F.Bulmer_ for the paper "_The boundary for quantum advantage in Gaussian boson sampling_".
  - `./src/gausstorch/utils/bcolors.py` contains the `bcolors` class used to color command line prints.
  - `./src/gausstorch/utils/display.py` contains functions used to either format strings/prints, or plot data.
  - `./src/gausstorch/utils/operations.py` contains functions frequently applied to PyTorch tensors.
    - Example: cholesky inverse for fast inversion of symmetric matrices
  - `./src/gausstorch/utils/param_processing.py` contains functions used to process `Qsyst` parameters.

## Testing

If you installed the requirements, the `pytest` package should be installed in your virtual environment. It will be used to test the source code.

The test module for each module `sample_module.py` is named `test_sample_module.py`, and is located in the same parent directory. It contains multiple functions which each test a function in `sample_module.py`.

For instance, you can check that the `torch_block()` function creates desired blocks and that `wigner()` returns a properly scaled wigner function, by running the command (from the project root directory):

```bash
pytest .\src\gausstorch\utils\test_operations.py
```

## Contact

If you have any questions, ask me a question at [juliendudas@gmail.com](juliendudas@gmail.com)
