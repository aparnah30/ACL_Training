s = "Hello, World! This is python"

a = {c.lower() for c in s if c.lower() in 'aeiou'}
print(a)