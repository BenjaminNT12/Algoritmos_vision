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

class Solution:
    # @param A : integer
    # @param B : long
    # @return a list of integers
    def findPerm(self, A, B):
        self.A = A
        self.B = B

        if(1 <= self.A and self.A <= 10**5):
            if  1 <= self.B and self.B <= min(10**18, math.factorial(self.A)):
                permutationA =  [i for i in range(1, self.A + 1)]
                permutationA = list(itertools.permutations(permutationA))
                print(permutationA)
                if sorted(permutationA) == permutationA:
                    return permutationA[self.B - 1]
            else:
                return ValueError("B is not valid")
        else:
            return ValueError("A is not valid")

def saltar_numeros(iteraciones, numeros_a_saltar):
    contador = 0
    numero_actual = 0

    while contador < iteraciones:
        numero_actual += 1

        if numero_actual in numeros_a_saltar:
            continue
        
        contador += 1
    return numero_actual
               

        
def main():
    A = 5
    B = 83
    print("Factorial of A: ", math.factorial(A))
    s = Solution()
    print("Objetivo", s.findPerm(A, B))
    print("numero de iteraciones", math.factorial(A))
    print("numero objetivo", B)
    multiplos = []
    for i in range(A):
        print("Factorial of N: ", math.factorial(A-i-1))
        multiplos.append(math.factorial(A-i-1))
    # print("Raiz N de A: ", math.factorial(A-1)*(B//math.factorial(A-1))+B%math.factorial(A-1)) #+B%math.factorial(A-1)
    # print("Division: ", B//math.factorial(A-1)+1)
    # print("Raiz N de A: ", math.factorial(A-1))
    
    # primer_valor = [x for x in range(1,math.factorial(A), multiplos[0])]
    # print("primer valor", primer_valor)
    
    maximo = B
    print(multiplos)
    # multiplo = []
    # multiplo.append(B//multiplos[0])
    # sumas = []
    # print("multiplicador1", multiplo[0], "Resultado", multiplos[0]*multiplo[0])
    # sumas.append(multiplos[0]*multiplo[0])
    
    # multiplo.append((maximo - sumas[0])//multiplos[1])
    # print("multiplicador1", multiplo[1], "Resultado1", multiplos[1]*multiplo[1])
    # sumas.append(multiplos[1]*multiplo[1])
    
    
    # multiplo.append((maximo - sumas[1] - sumas[0])//multiplos[2])
    # print("multiplicador1", multiplo[2], "Resultado1", multiplos[2]*multiplo[2])
    # sumas.append(multiplos[2]*multiplo[2])
    
    
    # multiplo.append((maximo - sumas[2] - sumas[1] - sumas[0])//multiplos[3])
    # print("multiplicador1", multiplo[3], "Resultado1", multiplos[3]*multiplo[3])
    # sumas.append(multiplos[3]*multiplo[3])
    
    multiplo2 = []
    acumulador = 0
    # multiplo2.append(B//multiplos[0])
    sumas2 = []
    for i in range(A):
        multiplo2.append((maximo - acumulador)//multiplos[i])
        print("multiplicador2", multiplo2[i], "Resultado2", multiplos[i]*multiplo2[i])
        sumas2.append(multiplos[i]*multiplo2[i])
        acumulador += sumas2[i]
        # print(sumas2[i])
        
    
    # print("multiplo", multiplo)
    # print("sumas", sumas)
    
    print("multiplo2", multiplo2)
    print("sumas2", sumas2)
    
    # contador = 0 
    # for k in range(math.factorial(A)):
    #     for i in range(A):
    #         for j in range(math.factorial(A-k-1)):
    #             contador += 1
    #             print("k: ", k , "contador: ", contador, "elements", i+1)
    #             if contador == math.factorial(A-k):
    #                 contador = 0
            

    # for i in range(math.factorial(A-2)):
    
    
    # count = 0
    
    # arreglo = []
    
    # arreglo.append(B//math.factorial(A-1)+1)
    # nivel = 0
    # for i in range(A):
    #     if i not in arreglo:
    #         nivel += 1
    #         for j in range(math.factorial(A-2)):
    #             print("j: ", j)
    #             if count < B%math.factorial(A-1):
    #                 count += 1
    #                 if count == B%math.factorial(A-1):
    #                     arreglo.append(i+1)
    #                     print("arreglo", arreglo)
    #                     # break
                    
                    
            
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()

    print("Tiempo: ",finish - start)    
    

    
    # 1-24
    # 2-6
    # 3-2
    # 4-1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
