import time
from collections import defaultdict
from tqdm.autonotebook import tqdm
from datetime import timedelta
from tensorboardX import SummaryWriter
from reward.utils import to_np
from reward.utils.memories import DefaultMemory

import numpy as np


class Logger:
    """
    Common logger used by all agents, aggregates values and print_table a nice table.

    Parameters
    ----------
    log_dir: str
        Path to write logs file.
    """

    def __init__(self, log_dir=None, log_freq=1, *, debug=False):
        self.log_dir = log_dir
        self.debug = debug
        self.log_freq = log_freq
        self.num_logs = 0
        # self.logs = DefaultMemory()
        # self.tf_logs = DefaultMemory()
        self.logs = dict()
        self.tf_logs = dict()
        self.precision = dict()
        self.tf_precision = dict()
        self.histograms = dict()
        self.time = time.time()
        self.steps_sum = 0
        self.eta = None
        self.time_header = None

        tqdm.write("Writing logs to: {}".format(log_dir))
        self.writer = SummaryWriter(log_dir=log_dir)

    def set_log_freq(self, log_freq):
        self.log_freq = log_freq

    def add_log(self, name, value, precision=2):
        """
        Register a value to a name, this function can be called
        multiple times and the values will be averaged when logging.

        Parameters
        ----------
        name: str
            Name displayed when printing the table.
        value: float
            Value to log.
        precision: int
            Decimal points displayed for the value (Default is 2).
        """
        self.logs[name] = value
        self.precision[name] = precision

    def add_tf_only_log(self, name, value, precision=2):
        """
        Register a value to a name, this function can be called
        multiple times and the values will be averaged when logging.
        Will not display the logs on the console but just write on the file.

        Parameters
        ----------
        name: str
            Name displayed when printing the table.
        value: float
            Value to log.
        precision: int
            Decimal points displayed for the value (Default is 2).
        """
        self.tf_logs[name] = value
        self.tf_precision[name] = precision

    def add_debug(self, name, value, precision=2):
        if self.debug: self.add_log(name, value, precision)

    def add_histogram(self, name, values):
        """
        Register a histogram that can be seen at tensorboard.

        Parameters
        ----------
        name: str
            Name displayed when printing the table.
        value: torch.Tensor
            Value to log.
        """
        self.histograms[name] = to_np(values)

    def log(self, step, header=None):
        """
        Use the aggregated values to print a table and write to the log file.

        Parameters
        ----------
        header: dict
            Optional header to include at the top of the table
            (Default is None, which show only the current step).
        """
        header = header or dict(Step=step)
        self.num_logs += 1
        # TODO: log_freq deprecated
        # if self.num_logs % self.log_freq == 0:
        # Take the mean of the values
        self.logs = {key: np.mean(value) for key, value in self.logs.items()}
        # Convert values to string, with defined precision
        avg_dict = {
            key: "{:.{prec}f}".format(value, prec=self.precision[key])
            for key, value in self.logs.items()
        }

        # Log to the console
        print_table(avg_dict, header)

        # Write tensorboard summary
        if self.writer is not None:
            self.tf_logs = {key: np.mean(value) for key, value in self.tf_logs.items()}
            for key, value in self.logs.items():
                self.writer.add_scalar(key, value, global_step=step)
            for key, value in self.tf_logs.items():
                self.writer.add_scalar(key, value, global_step=step)
            for key, value in self.histograms.items():
                self.writer.add_histogram(key, value, global_step=step)

        # Reset dict
        self.logs = dict()
        self.tf_logs = dict()
        self.histograms = dict()


def print_table(tags_and_values_dict, header=None, width=60):
    """
    Prints a pretty table =)
    Expects keys and values of dict to be a string
    """

    tags_maxlen = max(len(tag) for tag in tags_and_values_dict)
    values_maxlen = max(len(value) for value in tags_and_values_dict.values())

    max_width = max(width, tags_maxlen + values_maxlen)
    table = []

    table.append("")
    if header:
        head_items = ["{} {}".format(k, v) for k, v in header.items()]
        head_len = sum(len(s) for s in head_items)
        num_spaces = int(
            np.ceil(
                (max_width - head_len - len(head_items))
                / ((len(head_items) - 1) * 2 + 2)
            )
        )
        spaces = num_spaces * " "
        head = ("{spc}|{spc}".format(spc=spaces)).join(head_items)
        head = spaces + head + spaces
        table.append(head)

    table.append((2 + max_width) * "-")

    for tag, value in tags_and_values_dict.items():
        num_spaces = 2 + values_maxlen - len(value)
        string_right = "{:{n}}{}".format("|", value, n=num_spaces)
        num_spaces = 2 + max_width - len(tag) - len(string_right)
        table.append("".join((tag, " " * num_spaces, string_right)))

    table.append((2 + max_width) * "-")

    tqdm.write("\n".join(table))
