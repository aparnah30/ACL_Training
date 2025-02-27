def list_operations(list1, list2):
    sum_list1 = sum(list1)
    sum_list2 = sum(list2)
    
    intersection = list(set(list1) & set(list2))
    
    difference = list(set(list1) - set(list2))
    
    result = {
        "sum": sum_list1 + sum_list2,
        "intersection": intersection,
        "difference": difference
    }
    
    return result

list_a = [1, 2, 3, 4]
list_b = [3, 4, 5, 6]
result = list_operations(list_a, list_b)
print(result)
