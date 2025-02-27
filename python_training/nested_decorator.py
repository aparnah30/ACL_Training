import time

def logged_func(func):
    def wrapper(*args, **kwargs):
        with open('eg.txt', 'a') as file:
            file.write(f"{func.__name__} {args}{kwargs}")
        return func(*args, **kwargs)
    return wrapper

def cached_func(func):
    cache = {}
    def wrapper(*args, **kwargs):
        if (args, frozenset(kwargs.items())) in cache:
            return cache[(args, frozenset(kwargs.items()))]
        result = func(*args, **kwargs)
        cache[(args, frozenset(kwargs.items()))] = result
        return result
    return wrapper

@logged_func
@cached_func
def long_run(c, y):
    time.sleep(2)
    return c+y

print(long_run(4,5))