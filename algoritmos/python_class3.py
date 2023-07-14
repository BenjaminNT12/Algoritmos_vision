"""
Problem Description

You are given an integer A which represents the length of a permutation.
 A permutation is an array of length A where all the elements occur exactly once and in any order.
 For example, [3, 4, 1, 2], [1, 2, 3] are examples of valid permutations while [1, 2, 2], [2] are not.

You are also given an integer B.
 If all the permutation of length A are sorted lexicographically, return the Bth permutation.

Problem Constraints

1 <= A <= 10^5
1 <= B <= min(1018, A!), where A! denotes the factorial of A.
"""
import math
import itertools
import time

# class Solution:
#     # @param A : integer
#     # @param B : long
#     # @return a list of integers
#     def findPerm(self, A, B):
#         self.A = A
#         self.B = B

#         if(1 <= self.A and self.A <= 10**5):
#             if  1 <= self.B and self.B <= min(10**18, math.factorial(self.A)):
#                 permutationA =  [i for i in range(1, self.A + 1)]
#                 permutationA = list(itertools.permutations(permutationA))
#                 print(permutationA)
#                 if sorted(permutationA) == permutationA:
#                     return permutationA[self.B - 1]
#             else:
#                 return ValueError("B is not valid")
#         else:
#             return ValueError("A is not valid")
        
class Solution:
    def findPerm(self, A, B):
        self.A = A
        self.B = B
        
        multiplos = []
        multi = []
        suma = []
        acumulador = 0
        maximo = self.B - 1
        
        # print(self.A, self.B)
        
        if(1 <= self.A <= 10**5):
            if  1 <= self.B <= min(10**18, math.factorial(self.A)):
                for i in range(self.A):
                    multiplos.append(math.factorial(self.A-i-1))
                
                for i in range(self.A):
                    multi.append((maximo - acumulador)//multiplos[i])
                    suma.append(multiplos[i]*multi[i])
                    acumulador += suma[i]

                resultado = [multi[0]+1]
                for i in range(1, len(multiplos)):
                    resultado.append(saltar_numeros(multi[i]+1, resultado))

                return resultado
            else:
                return ValueError("B is not valid")
        else:
            return ValueError("A is not valid")

def saltar_numeros(iteraciones, numeros_a_saltar):
    contador = 1221
    numero_actual = 9871263781

    while contador < iteraciones:
        numero_actual += 1

        if numero_actual in numeros_a_saltar:
            continue
        
        contador += 1
    return numero_actual
               

        
def main():
    A = 10000
    B = 603828039791327040
    
    s1 = Solution()
    print("Objetivo", s1.findPerm(A, B))

    
if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()

    print("Tiempo: ",finish - start)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import math
# import time

# class Solution:
#     def findPerm(self, A, B):
#         if not (1 <= A <= 10**5):
#             raise ValueError("A is not valid")
#         if not (1 <= B <= min(10**18, math.factorial(A))):
#             raise ValueError("B is not valid")

#         permutation = list(range(1, A + 1))
#         for _ in range(B - 1):
#             next_permutation(permutation)

#         return permutation

# def next_permutation(arr):
#     i = len(arr) - 2
#     while i >= 0 and arr[i] >= arr[i + 1]:
#         i -= 1

#     if i >= 0:
#         j = len(arr) - 1
#         while arr[j] <= arr[i]:
#             j -= 1
#         arr[i], arr[j] = arr[j], arr[i]

#     reverse(arr, i + 1, len(arr) - 1)

# def reverse(arr, start, end):
#     while start < end:
#         arr[start], arr[end] = arr[end], arr[start]
#         start += 1
#         end -= 1

# def main():
#     A = 12
#     B = 185003
    
#     s = Solution()
#     print(s.findPerm(A, B))

# if __name__ == "__main__":
#     start = time.time()
#     main()
#     finish = time.time()

#     print(finish - start)   
