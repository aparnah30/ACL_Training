class Rectangle:
    def __init__(self, height, width):
        self._h = height
        self._w = width

    def add_area(self, r2):
        return self._h + self._w + r2._h + r2._w
    
    def scaling(self, factor):
        return self._h * factor, self._w * factor
    
r = Rectangle(5, 6)
r1 = Rectangle(7,8)
print(r.add_area(r1))
print(r.scaling(5))
        