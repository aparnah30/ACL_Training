def ret_sliced(s, i, j):
    return s[i:j]

num = [1, 2, 3, 4]
num2 = (1, 2, 3, 4)
num4 = "aparna"

a = ret_sliced(num, 2, 3)
b = ret_sliced(num2, 2, 3)
d = ret_sliced(num4, 2, 3)

print(a, b, d)
    


