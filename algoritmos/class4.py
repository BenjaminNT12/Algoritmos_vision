import math

class alumn():
    def __init__(self, name, branch):
        self.name = name
        self.branch = branch
        
    def printAlumno(self):
        print(self.name, self.branch)
        
class forms():
    def __init__(self, edge=0, radius=0):
        self.edge = edge
        self.radius = radius

    def square(self):
        return self.edge**2
    
    def triangle(self,altura,base):
        self.altura = altura
        self.base = base
        return self.altura*self.base*0.5
    
    def circle(self):
        return math.pi*self.radius**2

alumn1 = alumn("Benjamin", "Programacion")
alumn2 = alumn("Nicolas", "Programacion")

alumn1.printAlumno()
alumn2.printAlumno()


form1 = forms(4,2)
form2 = forms()
form3 = forms(5,radius=5)

print(f"Cuadrado {form1.square()}, triangle {form2.triangle(2,3)}, circulo {form3.circle()}")


class loginPage:
    def __init__(self, user, password):
        self.user = user
        self.password = password
        
    def login(self, salario):
        return f"user {self.user} --> {self.password}, salario: {salario}"
    
usuario = loginPage("Benjamin", "password")

print(usuario.login(60000))


def decorador(func):
    def camuflaje():
        fruta = func()
        return "Esto es un pay de " + fruta 
    return camuflaje 

@decorador
def fruta():
    return "Manzana"

# fruta()

print(f"fruta {fruta()}")

suma2numeros = lambda x, y: x + y

resta2numeros = lambda x,y: x-y

result = suma2numeros(2, 3)
result2 = resta2numeros(2, 3)

print(result)
print(result2)


my_list = [10,8,6,7,9,5,1,3,2,4]
print(sorted(my_list))
my_list = sorted(my_list)
print(my_list)
my_list = [10,8,6,7,9,5,1,3,2,4]
my_list.sort()
print(my_list)


numbers = [x for x in range(1,51)]

def sum2all(list):
    
    n/2+

print(numbers)

