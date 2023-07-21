import time

def sum(lista, n):
    total = 0
    for i in range(n):
        total += lista[i]
    
    return total

def main():
    lista = [12]*1000000000
    # print(lista)
    n = len(lista)
    
    start = time.time()
    res = sum(lista, n)
    finish = time.time()
    print(finish - start)
    print(res)
    
if __name__ == "__main__":
    main()
    
    