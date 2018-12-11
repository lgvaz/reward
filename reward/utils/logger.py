import numpy as np
from collections import namedtuple
from tqdm.autonotebook import tqdm
from tensorboardX import SummaryWriter
from reward.utils import to_np
from reward.utils.memories import DefaultMemory
from reward.utils import global_step


Log = namedtuple('Log', 'val prec hid')


class Logger:
    "Common logger used by all agents, writes to file and prints a pretty table."
    def __init__(self, logdir=None, logfreq=1000, maxsteps=None):
        self.logdir, self.logfreq, self.pbar = logdir, logfreq, tqdm(total=maxsteps, dynamic_ncols=True, unit_scale=True)
        self.logs, self.histograms, self.writer, self._next_log = {}, {}, SummaryWriter(log_dir=logdir), global_step.get() + logfreq
        global_step.subscribe_add(self._gstep_callback)
        tqdm.write("Writing logs to: {}".format(logdir))

    def add_log(self, name, value, precision=2, hidden=False, force=False):
        self.logs[name] = Log(val=value, prec=precision, hid=hidden)
        if force: self.writer.add_scalar(name, value, global_step=global_step.get())

    def add_histogram(self, name, values): self.histograms[name] = to_np(values)

    def log(self, header=None):
        step, rate = global_step.get(), self.pbar.n/(self.pbar._time() - self.pbar.start_t)
        header = header or dict(Step=step, Rate=f'{rate:.2f} steps/s')
        logs = {k: f'{v.val:.{v.prec}f}' for k, v in self.logs.items() if not v.hid}
        if logs: print_table(logs, header)
        self.add_log(name='steps_second', value=rate, hidden=True)
        for k, v in self.logs.items(): self.writer.add_scalar(k, v.val, global_step=step)
        for k, v in self.histograms.items(): self.writer.add_histogram(k, v, global_step=step)
        self.logs, self.histograms = {}, {}

    def close(self): self.pbar.close()

    def _gstep_callback(self, gstep):
        if self._next_log < gstep:
            self.log()
            self._next_log = gstep + self.logfreq
        self.pbar.update(gstep - self.pbar.n)


def print_table(tags_values, header=None, width=60):
    "Prints a pretty table =). Expects keys and values of dict to be a string"
    tags_maxlen = max(len(tag) for tag in tags_values)
    values_maxlen = max(len(value) for value in tags_values.values())
    width = max(width, tags_maxlen + values_maxlen)
    table = []
    table.append("")
    if header:
        head_items = ["{} {}".format(k, v) for k, v in header.items()]
        head_len = sum(len(s) for s in head_items)
        num_spaces = int(np.ceil((width - head_len - len(head_items)) / ((len(head_items) - 1) * 2 + 2)))
        spaces = num_spaces * " "
        head = ("{spc}|{spc}".format(spc=spaces)).join(head_items)
        head = spaces + head + spaces
        table.append(head)
    table.append((2 + width) * "-")
    for tag, value in tags_values.items():
        num_spaces = 2 + values_maxlen - len(value)
        string_right = "{:{n}}{}".format("|", value, n=num_spaces)
        num_spaces = 2 + width - len(tag) - len(string_right)
        table.append("".join((tag, " " * num_spaces, string_right)))
    table.append((2 + width) * "-")
    tqdm.write("\n".join(table))
