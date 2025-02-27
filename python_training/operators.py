from functools import reduce

def operate(num):
    def add(a, b):
        return a + b
    s = reduce(add, num)

    avg = s / len(num) if num else 0

    def max(a, b):
        return a if a>b else b
    m = reduce(max, num)

    return s, avg, m

num = [1,2,3,4,5]
a, b, c = operate(num)
print(a, b, c)