def get_age(a):
    if a < 0:
        raise ValueError("Age cannot be negative")
    if a > 150:
        raise OverflowError("Age cannot be greater than 150")
    return a

get_age(800)