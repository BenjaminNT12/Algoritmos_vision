def main():
    str_list = ['given', 'intern', 'InterviewBit', 'network', 'local', 'multiple', 'define', 'nodes', 'algorithm', 'allows', 'community', 'phase', 'single']
    my_list = [str_list[num] for num in range(len(str_list)) if len(str_list[num])%2 == 1]
    
    print(my_list)
    return 0

if __name__ == '__main__':
    main()