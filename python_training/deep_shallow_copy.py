import copy

def modify(lst):
    a = copy.copy(lst)
    b = copy.deepcopy(lst)
    a[2] = 0
    b[2] = 0
    return a, b, lst

num = [1, 3, 5, 7]
a, b, c = modify(num)
print("Shallow Copy: ", a)
print("Deep Copy: ", b)
print("Original List: ", c)
