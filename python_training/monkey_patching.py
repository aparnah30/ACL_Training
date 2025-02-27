class Person:
    def __init__(self, name):
        self.name = name

    def introduce(self):
        return f"Hey, I am {self.name}"
    
def new_introduce(self):
    return f"Bye, I am {self.name}"

p1 = Person("aparna")
print(p1.introduce())

Person.introduce = new_introduce

print(p1.introduce())
    