import re

pattern = r"\d+(?= \s*(USD |EUR))"
text = "The 30 quick brown fox jumps over the 20 lazy dog @ 5:00 pm in &23"
text1 = "The price is 100 USD"
text2 = "The total cost is 300 USD, which is the final price 200 EUR."

print(re.findall(pattern, text))
print(re.findall(pattern, text2))
