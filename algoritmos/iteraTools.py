# Python’s Itertool is a module that provides various 
# functions that work on iterators to produce complex 
# iterators. This module works as a fast, memory-efficient 
# tool that is used either by themselves or in 
# combination to form iterator algebra.

# Different types of iterators provided by this module 
# are Infinite Iterators, Combinatoric iterators and 
# Terminating iterators.

# Infinite Iterators

# Iterator in Python is any Python type that can be 
# used with a ‘for in loop’. Python lists, tuples, 
# dictionaries, and sets are all examples of inbuilt 
# iterators. But it is not necessary that an iterator
# object has to exhaust, sometimes it can be infinite.
# Such type of iterators are known as Infinite iterators.

# Python provides three types of infinite itertors:

# count(start, step): This iterator starts printing 
# from the “start” number and prints infinitely. If 
# steps are mentioned, the numbers are skipped else 
# step is 1 by default.

# See the below example for its use with for in loop.

# import itertools
# # for in loop
# for i in itertools.count(5, 5):
#     if i == 35:
#         break
#     else:
#         print(i, end =" ")
# # prints 5 10 15 20 25 30
# cycle(iterable): This iterator prints all values in 
# order from the passed container. It restarts printing 
# from the beginning again when all elements are printed 
# in a cyclic manner.

# import itertools

# count = 0

# # for in loop
# for i in itertools.cycle('AB'):
#     if count > 7:
#         break
#     else:
#         print(i, end = " ")
#         count += 1
# # prints A B A B A B A B
# repeat(val, num): This iterator repeatedly prints the 
# passed value infinite number of times. If the optional 
# keyword num is mentioned, then it repeatedly prints 
# num number of times.

# import itertools

# # using repeat() to repeatedly print number
# print ("Printing the numbers repeatedly : ")
# print (list(itertools.repeat(25, 4)))
# # prints
# Printing the numbers repeatedly :
# [25, 25, 25, 25]
# Try the following excercise in the editor below.

# Perform the operations as described in the comments 
# in the order given.
import itertools


def main():
    # print 1000 space separated integers starting 
    # from 1000 with common difference 500
    # 1000 1500 2000 2500 3000........
    # There should be exactly one space after every 
    # integer
    count = 0
    for num in itertools.count(1000,500):
        if count == 1000:
            break
        else:
            print(num, end = " ")
            count += 1
    
    print()
    # print all uppercase alphabets 15 times, printing 
    # from A-Z then repeating again
    # A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A B C D........
    # There should be exactly one space after every character
    letters = itertools.cycle(chr(char) for char in range(ord('A'), ord('Z')+1))

    print(*list(next(letters) for _ in range(15*26)), sep=' ')
    # print list of integers containing 1000 4's
    
    print(list(itertools.repeat(4, 1000)))

    return 0


if __name__ == '__main__':
    main()
