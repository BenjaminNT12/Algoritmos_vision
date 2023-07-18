import math
import statistics as stat

class Solution:
    def majorityElement(self, _A):
        self.A = _A
        
        majority = stat.mode(self.A) 
        count = 0
        
        for i in range(len(self.A)):
            if majority == self.A[i]:
                count += 1
        
        majorityGoal = math.floor(len(self.A) / 2)
        
        if count > majorityGoal:
            return majority
            

def main():
    ma = Solution()
    A = [ 1, 1, 1, 2, 2 ]
    
    print("Majority: ", ma.majorityElement(A))
    
if __name__ == "__main__":
    main()