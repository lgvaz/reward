CONFIG = {'gstep': 0}
CBACK = dict(add=[], set=[])


def get(): return CONFIG['gstep']
def reset(): set(v=0)

def add(v): 
    CONFIG['gstep'] += v
    for cback in CBACK['add']: cback(CONFIG['gstep'])

def set(v):
    CONFIG['gstep'] = v
    for cback in CBACK['set']: cback(CONFIG['gstep'])

def subscribe_add(callback): CBACK['add'].append(callback)
def subscribe_set(callback): CBACK['set'].append(callback)