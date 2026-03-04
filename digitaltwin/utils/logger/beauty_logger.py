import os


class BeautyLogger:
    """
    Lightweight logger.
    """

    def __init__(self, log_path: str, log_name: str = 'xxx.log', verbose: bool = True):
        """
        Lightweight logger.

        Example::

            >>> from digitaltwin.utils.logger import BeautyLogger
            >>> logger = BeautyLogger(log_path=".", log_name="xxx.log", verbose=True)

        :param log_path: the path for saving the log file
        :param log_name: the name of the log file
        :param verbose: whether to print the log to the console
        """
        self.log_path = log_path
        self.log_name = log_name
        self.verbose = verbose

    def _write_log(self, content, type):
        with open(os.path.join(self.log_path, self.log_name), "a") as f:
            f.write("[Digital Twin:{}] {}\n".format(type.upper(), content))

    def warning(self, content, local_verbose=True):
        """
        Print the warning message.

        Example::

            >>> logger.warning("This is a warning message.")

        :param content: the content of the warning message
        :param local_verbose: whether to print the warning message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="warning")
        self._write_log(content, type="warning")

    def module(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.module("This is a module message.")

        :param content: the content of the module message
        :param local_verbose: whether to print the module message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="module")
        self._write_log(content, type="module")

    def info(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.info("This is a info message.")

        :param content: the content of the info message
        :param local_verbose: whether to print the info message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="info")
        self._write_log(content, type="info")


def beauty_print(content, level=None, type=None):
    """
    Print the content with different colors.

    Example::

        >>> import digitaltwin as dt
        >>> dt.logger.beauty_print("This is a warning message.", type="warning")

    :param content: the content to be printed
    :param level: support 0-3, 0 for error and warning, 1 for module, 2 for info
    :param type: support "warning", "module", "info"
    :return:
    """
    if level is None and type is None:
        level = 3
    if level == 0 or type == "warning":
        print("\033[1;31m[WARNING] {}\033[0m".format(content))  # For error and warning (red)
    elif level == 1 or type == "module":
        print("\033[1;33m[MODULE] {}\033[0m".format(content))  # start of a new module (light yellow)
    elif level == 2 or type == "info":
        print("\033[1;35m[INFO] {}\033[0m".format(content))  # start of a new function (light purple)
    elif level == 3:
        print("\033[1;36m{}\033[0m".format(content))  # For mentioning the start of a new class (light cyan)
    elif level == 4:
        print("\033[1;32m{}\033[0m".format(content))  # For mentioning the start of a new method (light green)
    elif level == 5:
        print("\033[1;34m{}\033[0m".format(content))  # For mentioning the start of a new line (light blue)
    else:
        raise ValueError("Invalid level")
