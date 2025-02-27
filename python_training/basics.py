#import this

def add(a, b):
    return a + b

print("sum of two numbers: ", add(5, 5))

def string_len(s):
    return len(s)

print("length of string: ", string_len("aparna"))

def is_prime(a):
    for i in range(3,a):
        if a % i == 0:
            return "No"
        
    return "Yes"

print("Is the number prime: ", is_prime(13))