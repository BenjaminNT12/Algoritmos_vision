# https://www.freecodecamp.org/learn/scientific-computing-with-python/python-for-everybody/introduction-why-program
# https://www.codecademy.com/catalog
# https://practice.geeksforgeeks.org/explore?page=1&sprint=a663236c31453b969852f9ea22507634&sortBy=submissions&sprint_name=SDE%20Sheet&utm_medium=newui_home&utm_campaign=first_section
# https://www.hackerearth.com/practice/
# https://www.interviewbit.com/practice/
# https://www.w3schools.com/python/python_numbers.asp
# https://structy.net/purchase
# https://leetcode.com/problemset/algorithms/


class my_class:
    x = 5
    
p1 = my_class()

print(p1.x)

class person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
p1 = person("John", 36)

print(p1.name)
print(p1.age)        

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def __str__(self):
        return f"Name: {self.name}, Age: {self.age}"
    
p1 = Person("John", 36)

print(p1)

class person1:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def myfunc(self):
        print("Hello my name is " + self.name + " and I am " + str(self.age) + " years old.")
        
p1 = person1("John", 36)

p1.myfunc()

p1.age = 40
p1.myfunc()


p1.myfunc()

class Person1:
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname
        
    def printname(self):
        print(self.firstname, self.lastname)
    
p1 = Person1("John", "Doe")

p1.printname()


# class Student(Person1):
#     pass

# p2 = Student("Mike", "Olsen")

# p2.printname()

class Student(Person1):
    def __init__(self, fname, lname):
        super().__init__(fname, lname)
    
    def printStudent(self):
        print(self.firstname, self.lastname)
        
p2 = Student("Mike", "Olsen")


p2.printStudent()