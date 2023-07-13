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
                print(math.factorial(self.A))
                permutationA = list(itertools.permutations(permutationA))
                print("aqui1")
                # print(permutationA)
                if sorted(permutationA) == permutationA:
                    # return permutationA[self.B - 1]
                    print("aqui")
            else:
                return ValueError("B is not valid")
        else:
            return ValueError("A is not valid")
        
def main():
    A = 11
    B = 185003
    
    s = Solution()
    print(s.findPerm(A, B))

if __name__ == "__main__":
    main()