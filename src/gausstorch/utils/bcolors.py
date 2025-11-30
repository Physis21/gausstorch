class bcolors:
    """Class containing colors for command line prints, using ANSI escape sequences.
    Source code taken from `stackoverflow <https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal>`_.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'