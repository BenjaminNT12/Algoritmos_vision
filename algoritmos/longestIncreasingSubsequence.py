# class Solution():
#     def lis(self, _A):
#         self.A = _A
        
# # def longest_increasing_subsequence(A):
#         n = len(self.A)
#         if n == 0:
#             return 0
        
#         dp = [1] * n
#         for i in range(1, n):
#             for j in range(i):
#                 if self.A[i] > self.A[j]:
#                     dp[i] = max(dp[i], dp[j] + 1)
#         return max(dp)
class Solution():
    def lis(self, _A):
        self.A = _A
        
        n = len(self.A)
        lista = []
        dp = [1] * n
        contador = 0
        lista.append(self.A[0])
        for i in range(1, len(self.A)):
            for j in range(i):
                print("i=", i, "j=", j,">>>>>", self.A[i]," > ", self.A[j])
                if self.A[i] > self.A[j]:
                    contador += 1
                    lista.append(self.A[j])
                    dp[i] = max(dp[i], dp[j] + 1)
                    print("Si paso: i= ", i, "j=", j,"---->", self.A[i]," > ", self.A[j], "-----> contador: ", contador, "dp",i,": ", dp[i])
            lista.append(self.A[i])
            contador = 0
            # print("lista", lista)
            print("self.A", self.A)
            lista[:] = []
        # print("dp: ", dp)
        #     if self.A[i] > lista[count]:
        #         count += 1
        #         lista.append(self.A[i])
        #         # print(f"En la {i} iteracion, agrege un valor nuevo, el cual es: {self.A[i]}")
        #     elif self.A[i] < lista[count] and self.A[i] > lista[count-1]:
        #         # print(f"En la {i} iteracion, modifique un valor, {lista[count]} por {self.A[i]}")
        #         lista[count] = self.A[i]
        # return len(lista)

def main():
    l1 = Solution()
    # A = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]

    A = [ 69, 54, 19, 51, 16, 54, 64, 89, 72, 40, 31, 43, 1, 11, 82, 65, 75, 67, 25, 
          98, 31, 77, 55, 88, 85, 76, 35, 101, 44, 74, 29, 94, 72, 39, 20, 24, 23, 66, 
          16, 95, 5, 17, 54, 89, 93, 10, 7, 88, 68, 10, 11, 22, 25, 50, 18, 59, 79, 87, 
          7, 49, 26, 96, 27, 19, 67, 35, 50, 10, 6, 48, 38, 28, 66, 94, 60, 27, 76, 4, 
          43, 66, 14, 8, 78, 72, 21, 56, 34, 90, 89 ]

    print(l1.lis(A))
    

if __name__ == "__main__":
    main()  
    
    
    
    
    
    
    
    # class Solution():
    # def lis(self, _A):
    #     self.A = _A
        
    #     print(self.A)
        
    #     lista = []
    #     count = 0
    #     _max = 0
    #     indices = []
    #     # lista.append(self.A[0])
    #     for k in range(len(self.A)):
    #         lista.append(self.A[k])
    #         # indices.append(k)
    #         for i in range( len(self.A)):
    #             if self.A[i] > max(lista):
    #                 # count += 1
    #                 lista.append(self.A[i])
    #                 # indices.append(i)
    #                 # print(f"En la {i} iteracion, agrege un valor nuevo, el cual es: {self.A[i]}")
    #             # elif self.A[i] < lista[count]:# and self.A[i] > lista[count-1]:
    #                 # print(f"En la {i} iteracion, modifique un valor, {lista[count]} por {self.A[i]}")
    #                 # lista[count] = self.A[i]
    #                 # indices[count] = i
    #             # else:
    #             #     print(f"En la {i} iteracion, no hice nada")
    #         # print("antes",lista)
    #         # for j in range(k, -1, -1):
    #             # print(self.A[j], lista[0])
                
    #             # if self.A[j] < lista[0]:
    #             #     lista.insert(0, self.A[j])
    #                 # indices.insert(0, j)
    #             # elif :
    #         if _max < len(lista):
    #             _max = len(lista)
    #         # print("lista: ", self.A)
    #         print("despues",lista)
    #         print("indices", indices)
    #         print("max", _max)
            
    #         lista[:] = []
    #         count = 0
            
        
    #     return len(lista)
        