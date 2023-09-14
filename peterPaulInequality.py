import numpy as np

def peter_paul_inequality_minus(a, b, c):
  """Returns True if a - b >= c, False otherwise."""
  return a - b >= c

# Check if a - b >= c
a = 10
b = 5
c = 3

if peter_paul_inequality_minus(a, b, c):
  print("a - b >= c")
else:
  print("a - b < c")