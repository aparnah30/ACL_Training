lst = [("aparna", 15), ("sayali", 24), ("om", 22), ("harshad" ,18), ("sakshi", 20)]

a = {name: age for name, age in lst if age > 18}
print(a)