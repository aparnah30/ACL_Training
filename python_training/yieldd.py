# def chunk_data(lst, chunk_size):
#     iterat = iter(lst)
#     for i in range(0, len(lst), chunk_size):
#         yield lst[i : i + chunk_size]
        
# for chunk in chunk_data([1,2,3,4,5,6,7,8], 2):
#     print(chunk)

def get_num(lst):
    iterat = iter(lst)
    for i in range(len(lst)):
        yield lst[i]

l=[2,4,5,7]
a = get_num(l)
for i in get_num(l):
    print(i)