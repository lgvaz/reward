class Space:
    def __init__(self, sz, order='chw'):
        assert order == 'chw', f'Only support order chw, got {order}'
        self.sz, self.order = sz, order