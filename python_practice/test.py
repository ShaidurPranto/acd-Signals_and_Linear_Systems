def functionHello(count,name) :
    "this function is a demo function"
    for x in range(count) :
        print(f"my name is {name}")

def functionKeyword(str) :
    "this function can be called like keywords"
    print(str)

def variableLenFunction(name,*var) :
    "this function can take variable lengths of arguments"
    print(name)
    for f in range(len(var)) :
        print(var[f])

def defaultArgumentFunction(name , age = 30) :
    print("name is ",name)
    print("age is ",age)

def multipleVal() :
    str = "what is your name"
    x = 55
    return str,x

a = ("hello",33)
b,c = a
print(b)
print(c)
print(a)