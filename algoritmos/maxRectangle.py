# Given a 2D binary matrix filled with 0’s and 1’s, 
# find the largest rectangle containing all ones 
# and return its area.

# Bonus if you can solve it in O(n^2) or less.

# Example :

# A : [  1 1 1
#        0 1 1
#        1 0 0 
#     ]

# Output : 4 

# As the max area rectangle is created by the 2x2 rectangle 
# created by (0,1), (0,2), (1,1) and (1,2)
import numpy as np

class Solution:
    # @param A : list of list of integers
	# @return an integer
    def maximalRectangle(self, _A):
        
        self.A = _A
        
        if not self.A:
            return 0

        rows = len(self.A)
        cols = len(self.A[0])
        print(rows, cols)
        histogram = [0] * cols
        # print(histogram)
        max_area = 0

        for i in range(rows):
            for j in range(cols):
                if self.A[i][j] == 0:
                    histogram[j] = 0
                else:
                    histogram[j] += 1
            max_area = max(max_area, largestRectangleArea(histogram))
            print(histogram,"Size of histogram: ")
            print("max_area: ", max_area)

        return max_area


def largestRectangleArea(heights):
    stack = []
    max_area = 0
    i = 0
    
    while i < len(heights):
        if not stack or heights[i] >= heights[stack[-1]]:
            stack.append(i)
            i += 1
        else:
            top = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, heights[top] * width)
    
    while stack:
        top = stack.pop()
        width = i if not stack else i - stack[-1] - 1
        max_area = max(max_area, heights[top] * width)
    
    return max_area


# Example usage:
# matrix = [
#     [1, 1, 1],
#     [0, 1, 1],
#     [1, 0, 0],
# ]

# print(maximalRectangle(matrix))  # Output: 4
                
def main():
    A = [[1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]]
    
    Hola = "Hola"
    print("Hola", Hola.find('a'))
    
    A = [[0]]
    s = Solution()
    # s.maximalRectangle(A)
    print(s.maximalRectangle(A))
    # print("xor: ", detect_rectangle(A))
    
if __name__ == "__main__":
    main()