def parse_n_divide(s1, s2):
    try:
        a = int(s1)
        b = int(s2)
        res = a/b
    except ZeroDivisionError:
        print("Cannot divide by zero")
    except:
        print("error occured")
    else:
        print("Result: ", res)
    finally:
        "Operation successful!"

parse_n_divide('9', '3')
parse_n_divide('9', '0')

