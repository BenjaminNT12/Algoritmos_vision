from collections import defaultdict
# name = "Benjamin"
name = "Nicolas"
print(f"Hola {name}")

salary = 60001
isMarried = True

print(f"mi salario es {salary}, es un buen salario")
print("mi salario es " + str(salary))
print("mi salario es ",salary, "es un buen salario")
print(type(name))
print(type(salary))
# for i in range(2,10,3):
#     print(i)
        
if name == "Benjamin":
    print("yes!!")
else:
    print("I don't know how is he!!")

if name == "Benjamin":
    print("yes!!")
elif name == "Nicolas2":
    print(f"I don't know how is {name}!!")
else:
    print("I don't know how is she!!")
    
    
if 5000<=salary<=60000:
    print("good salay")
else: 
    print("isn't a good salay") 
    
if isMarried is True: print("That's sounds very well")

browsers = ["chrome", "firefow", "safari"]
browsers2 = ("chrome", "firefow", "safari")
browsers3 = {"chrome":[] ,
             "firefow": "enough",
             "safari":"better" 
             }

browsers.append("Opera")
print(browsers[3])

print(type(browsers))
for browser in browsers:
    print(f"el, es {browser}")
    
# browsers2
print(type(browsers2))
for browser in browsers2:
    print(f"el, es {browser}")
    
browsers3["chrome"].append("exist")
print(type(browsers3))
for browser in browsers3:
    print(f"el, es {browser},== {browsers3[browser]}")
    
print(browsers3.get("firefow"))
print(browsers[3])


browsers.insert(5, "chromium")
print(browsers)
browsers.remove("firefow")
print(browsers)
browsers.insert(1, "firefox")
print(browsers.sort())


for browser in browsers3.values():
    print(f"el, es {browser}")
    
    
listOne2Ten = [x for x in range(1,11,2)]


print(listOne2Ten)

browsers3.pop("firefow")
browsers3["firefox"]="good"

dict2 = {"ferrari":"best", "redBull":"better"}

browsers3.update(dict2)

print(browsers3)


if "ferrari" in browsers3:
    print("eso no deberia ir ahi")
else:
    print("todo en orden")

def multiply2numbers(num1=0, num2=0):
    if num1 != 0 and num2 != 0:
        return num1*num2
    
print(multiply2numbers(3,4))
