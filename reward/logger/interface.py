from .logger import Logger


_logger = Logger()

def set_logdir(logdir): return _logger.set_logdir(logdir)
def set_logfreq(logfreq): return _logger.set_logfreq(logfreq)
def set_maxsteps(maxsteps): return _logger.set_maxsteps(maxsteps=maxsteps)
def set_debug(debug=True): _logger.debug = debug

def log(): return _logger.log()

def add_histogram(name, values): return _logger.add_histogram(name=name, values=values)

def add_log(name, value, precision=2, hidden=False, force=False):
    return _logger.add_log(name=name, value=value, precision=precision, hidden=hidden, force=force)

def add_header(name, value): return _logger.add_header(name=name, value=value)

def is_debug(): return _logger.debug