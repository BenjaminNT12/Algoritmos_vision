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
            prime = False
    if n%1 == 0 and n%n == 0 and prime == True and n > 1:
        return True
    else:
        return False
    

print("Max val: ", max_value([4, 7, 2, 8, 10, 9]))
print("Is prime", is_prime(1))