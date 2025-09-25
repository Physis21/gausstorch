# Source repository for 'gausstorch', a package allowing the simulation of coupled bosonic modes, with PyTorch

## Requirements & installing

- Have Python **version 3.12.1** installed on your computer
- First download the GitLab code
- From the command line, move into the root folder of the downloaded code

```console
cd root_folder
```

- create a virtual environment from Pycharm or using the command (!!! use the correct Python 3.12.1 version)

```console
python -m venv .venv
```

- Activate the virtual environment

```console
source .\venv/bin/activate # on Unix/macOS
.\.venv\Scripts\activate  # on Windows
```

- from the virtual environment run the command:

```console
pip install -r requirements.txt  # install pip packages used in src/
pip install --editable .  # install the src package
```

- The 'gausstorch' package is now usable from the package `gausstorch`

## READ THE TUTORIAL
Go to [tutorials/qsyst.ipynb](tutorials/qsyst.ipynb) to see how the **Qsyst** class is used to model coupled bosonic modes.


## Structure of src/gausstorch

- `./src/gausstorch/constants`: constants reused throughout the package.
  - example: **SIM_DATA_DIR_PATH** defines the path where the user wants simulation data to be saved.
- `./src/gausstorch/libs/qsyst`: contains the **Qsyst** class. It contains methods to simulate an arbitrary number of coupled bosonic modes.
  - On initialization, an instance of **Qsyst** requires an **init_pars** argument, which specifies the drive, coupling and dissipation parameters within the modes.
  - Learning models using Gaussian dynamics should be child classes of **Qsyst**.
- `./src/gausstorch/utils`: very short and specific functions used throughout functions in `./src/gausstorch/libs`
  - `./src/gausstorch/utils/_loop_hafnian_subroutines.py` and `loop_hafnian.py`are used to compute **loop hafnians**. The functions are adapted to PyTorch from the github repository <https://github.com/jakeffbulmer/gbs> , made by _Jacob F.F.Bulmer_ for the paper "The boundary for quantum advantage in Gaussian boson sampling".
  - `./src/gausstorch/utils/bcolors.py` contains the **bcolors** class used to color command line prints.
  - `./src/gausstorch/utils/display.py` contains functions used to either format strings/prints, or plot data.
  - `./src/gausstorch/utils/operations.py` contains functions frequently applied to PyTorch tensors.
    - Example: cholesky inverse for fast inversion of symmetric matrices
  - `./src/gausstorch/utils/param_processing.py` contains functions used to process **Qsyst** parameters.
