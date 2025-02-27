class Employee:

    min_sal = 3000

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    @staticmethod
    def valid_sal(self, salary):
        return salary >= Employee.min_sal


    @classmethod
    def create_employee(cls, name, salary):
        if cls.valid_sal(salary):
            return cls(name, salary)
        else:
            raise ValueError("salary less than 3000")
    
    def __repr__(self):
        return f"Name of the emp is {self.name}, salary is {self.salary}"
    
e1 = Employee("aparna", 3000)
print(e1)
    
e2 = Employee("sayali", 2000)
print(e2)


