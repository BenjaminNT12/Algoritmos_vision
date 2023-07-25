class Student:
    def __init__(self, name, branch):
        self.name = name
        self.branch = branch
    
    def printname(self):
        print(self.name, self.branch)
        
        
def main():    
    stud1 = Student("Robin", "CSE") 
    stud2 = Student("Rahul", "ECE") 
    
    stud1.printname()
    stud2.printname()
    print("Nombre accesando directamente: ", stud1.name)
    
if __name__ == "__main__":
    main()