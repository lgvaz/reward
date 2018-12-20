import numpy as np, reward.utils as U
from collections import namedtuple, OrderedDict
from tqdm.autonotebook import tqdm
from tensorboardX import SummaryWriter


Log = namedtuple('Log', 'val prec hid')
class Logger:
    "Common logger used by all agents, writes to file and prints a pretty table."
    def __init__(self, logfreq=1000, maxsteps=None):
        self.logfreq, self.pbar = int(logfreq), tqdm(total=maxsteps, dynamic_ncols=True, unit_scale=True)
        self.logs,self.hists,self.header,self.writer,self._next_log = {},{},OrderedDict(),None,U.global_step.get()+logfreq
        self.debug = False
        U.global_step.subscribe_add(self._gstep_callback)

    def set_logdir(self, logdir): self.writer = SummaryWriter(log_dir=logdir)
    def set_logfreq(self, logfreq): self.logfreq = logfreq
    def set_maxsteps(self, maxsteps):
        self.maxsteps = maxsteps
        self.pbar.total = maxsteps

    def add_log(self, name, value, precision=2, hidden=False, force=False):
        self._check_writer()
        self.logs[name] = Log(val=value, prec=precision, hid=hidden)
        if force: self.writer.add_scalar(name, U.to_np(value), global_step=U.global_step.get())

    def add_histogram(self, name, values): self.hists[name] = U.to_np(values)

    def add_header(self, name, value): self.header[name] = value

    def log(self):
        self._check_writer()
        step, rate = U.global_step.get(), self.pbar.n/(self.pbar._time() - self.pbar.start_t)
        self.header.update(OrderedDict(Step=step, Rate=f'{rate:.2f} steps/s'))
        for k, v in self.logs.items(): self.logs[k] = self.logs[k]._replace(val=U.to_np(self.logs[k].val))
        logs = {k: f'{v.val:.{v.prec}f}' for k, v in self.logs.items() if not v.hid}
        if logs: print_table(logs, self.header)
        self.add_log(name='steps_second', value=rate, hidden=True)
        for k, v in self.logs.items(): self.writer.add_scalar(k, v.val, global_step=step)
        for k, v in self.hists.items(): self.writer.add_histogram(k, v, global_step=step)
        self.logs, self.hists, self.header = {}, {}, OrderedDict()

    def close_pbar(self): self.pbar.close()

    def _gstep_callback(self, gstep):
        if self._next_log < gstep:
            self.log()
            self._next_log = gstep + self.logfreq
        self.pbar.update(gstep - self.pbar.n)

    def _check_writer(self):
        if self.writer is None: self.writer = SummaryWriter(log_dir='/tmp/reward/logs')


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
