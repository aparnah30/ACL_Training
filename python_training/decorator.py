import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        s = time.time()
        res = func(*args, **kwargs)
        e = time.time()
        times = e - s
        print(f"Time required: {times}")
        return res
    return wrapper

@measure_time
def long_run():
    time.sleep(3)

long_run()
