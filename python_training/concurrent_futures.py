import concurrent.futures
import math
import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    sqr = math.floor(math.sqrt(n))
    for i in range(3, sqr, 2):
        if n % i == 0:
            return False
        
    return True

def get_time(s, e):
    return e - s

primes = [3, 2, 19, 6, 2977453]

def main():
    start = time.time()

    #without multiprocessing
    for n, result in zip(primes, map(is_prime, primes)):
        print(f"{n} is prime: {result}")

    #with multiprocessing
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for n, result in zip(primes, map(is_prime, primes)):
    #         print(f"{n} is prime: {result}")


    end = time.time()
    t = get_time(start, end)
    print(t)

main()

