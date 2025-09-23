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
