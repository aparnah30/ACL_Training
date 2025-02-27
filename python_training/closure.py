def outer(x):
    def inner(y):
        return x + y
    return inner

c = outer(10)
print(c(20))