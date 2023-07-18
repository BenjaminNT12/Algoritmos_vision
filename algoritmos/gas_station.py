class Solution:
    def canCompleteCircuit(self, _A, _B):
        self.A = _A
        self.B = _B
        
        if sum(self.A) < sum(self.B):
            return -1
        start = 0
        tank = 0
        print(self.A)
        for i in range(len(self.A)):
            tank += self.A[i] - self.B[i]
            if tank < 0:
                start = i + 1
                tank = 0
        return start

def main():
    c1 = Solution()
    A = [1, 2]
    B = [2, 1]
    print("Indice minimo: " ,c1.canCompleteCircuit(A, B))
    
if __name__ == "__main__":
    main()
    
    
    