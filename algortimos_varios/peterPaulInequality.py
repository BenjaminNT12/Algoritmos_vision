import numpy as np

def peter_paul_inequality_minus(a, b):
  """Returns True if a - b >= c, False otherwise."""
  print(f"AB = {-a*b}")
  print(f"(a^2)/2 + (b^2)/2 = {((a**2)/2 + (b**2)/2)}" )
  return -a*b <= ((a**2)/2 + (b**2)/2)

# Check if a - b >= c
a = -1
b = -1


if peter_paul_inequality_minus(a, b):
  print("a*b <= (a - b)")
else:
  print("No se cumple")