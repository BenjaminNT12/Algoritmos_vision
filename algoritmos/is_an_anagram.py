import time


# Given wo strings s1 and s2, check if theys are anagrams of each other.
# they're made of the same characters but in different order.


def are_an_anagram(word1, word2):
    return sorted(word1) == sorted(word2)

print(are_an_anagram("nameless", "salesmen"))


# Given a sorted of integer arr and an integer target, find the index 
# of the first and last position of target in arr. In target can't be 
# found in arr, return [-1, -1].

# input = [2, 5, 5, 5, 5, 5, 5, 7, 9, 9]
# target = 5
# output = [1, 6]


def find_index(sorter_arr, targ):
    start = time.time()
    first = 0
    
    if targ not in sorter_arr:
        return -1, -1
    
    while first < len(sorter_arr) and sorter_arr[first] != targ:
        first += 1
    
    last = first
    
    while last < len(sorter_arr) and sorter_arr[last] == targ:
        last += 1
        
    last -= 1
    
    finish = time.time()
    
    print(finish - start)
    
    return first, last
        
arr = [2, 1, 5, 25, 15, 35, 25, 7, 9, 9]
target = 5

print(find_index(arr, target))  

def largest_kth_element(arr, k):
    start = time.time()
    
    arr.sort(reverse=True)
    
    return arr[:k+1]

arr = [4, 2, 9, 7, 5, 6, 7, 1, 3]
k = 4
print(largest_kth_element(arr, 3))

# arr = [elem for elem in arr]

print(arr)

    

    