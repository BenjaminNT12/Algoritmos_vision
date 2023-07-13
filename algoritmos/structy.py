from math import sqrt, floor
import time

def max_value(nums):
  
  max_val = nums[0]
  for num in range(len(nums)):
    if nums[num] > max_val:
      max_val = nums[num]
      
  return max_val
    
def is_prime(n):
    prime = True
    
    for i in range(2 ,n+1):
        if n%i == 0 and i < n:
            return  False
    if n%1 == 0 and n%n == 0 and n > 1:
        return True
    else:
        return False

def is_prime_new(n):
  if n < 2:
    return False
  
  for i in range(2, floor(sqrt(n)) + 1):
    # print(i)
    # print(floor(sqrt(n)) + 1)
    if n % i == 0:
      return False
    
  return True    

print("Max val: ", max_value([4, 7, 2, 8, 10, 9]))

start = time.time()
print("Is prime", is_prime(524287))
finish = time.time()
print("Time taken: ", finish - start)

start = time.time()
print("Is prime", is_prime_new(524287))
finish = time.time()
print("Time taken: ", finish - start)

print(floor(sqrt(11)) + 1)
print(sqrt(11))