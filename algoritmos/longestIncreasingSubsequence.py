class Solution():
    def lis(self, _A):
        self.A = _A
        
        lista = []
        lista.append(min(self.A[:]))
        print(lista)
        count = 0
        possiblesSeq = [] 
        
        for k in range(len(self.A)):
            for i in range(len(self.A)):
                if lista[count] < min(self.A[i:]):
                    count += 1
                    lista.append(min(self.A[i:]))
            possiblesSeq.append(len(lista))
        print(possiblesSeq)
        print(lista)
        # print(len(lista))

def main():
    l1 = Solution()
    A = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    print(l1.lis(A))
    

if __name__ == "__main__":
    main()  
    
    
    