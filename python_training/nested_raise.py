class CustomValidationError(Exception):
    def __init__(self, message="Invalid input value."):
        self.message = message
        super().__init__(self.message)

class CustomTypeError(Exception):
    def __init__(self, message="Input must be an integer."):
        self.message = message
        super().__init__(self.message)


def validate_input(num):
    try:
        if not isinstance(num, int):  
            raise TypeError("Input is not an integer.")  
        if num < 0:  
            raise ValueError("Negative integer is not allowed.")  
        
    except ValueError as ve:
        raise CustomValidationError("A negative number was entered.") from ve
    except TypeError as te:
        raise CustomTypeError("The input is not an integer.") from te

try:
    validate_input(5) 
    print("Valid input")
except (CustomValidationError, CustomTypeError) as e:
    print(f"Error: {e}")

try:
    validate_input(-10)
except (CustomValidationError, CustomTypeError) as e:
    print(f"Error: {e}")

try:
    validate_input("hello")
except (CustomValidationError, CustomTypeError) as e:
    print(f"Error: {e}")
