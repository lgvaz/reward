import time


def timeit(func):
    name = func.__name__

    def exec(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print("\n{} took {:.3f} us".format(name, (end - start) * 1e6))
        return result

    return exec
