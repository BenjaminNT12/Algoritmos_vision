def main():
    my_set = set([1, 3, 2, 4, 1, 3, 3, 0])
    
    # add 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 to my_set
    add_num = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22, 23]
    
    for num in add_num:
        my_set.add(num)
    
    # my_set.add(10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    
    # delete 2 and 3 from my_set
    
    del_num = [2, 3]
    
    for num in del_num:
        my_set.remove(num)
    
    # my_set.remove(2,3)
    
    li = list(my_set)
    li.sort()

    print(li)
    return 0

if __name__ == '__main__':
    main()