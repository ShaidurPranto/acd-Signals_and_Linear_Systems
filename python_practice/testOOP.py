class Employee :
    # attributes created outside the constructor , are known as class attributes 
    king_name = "pranto"
    def __init__(self,name,age) :
        # attributes created inside the init method are instance attributes
        self.name = name
        self.age = age

    def printInfo(self) :
        print("the employee name is : ",self.name , " and age is : ",self.age)

    def __str__(self) :
        return f"Name : {self.name} , age : {self.age}"

a = Employee("pranto",22)
a.printInfo()
print(a.king_name)

b = Employee("vodor",33)
b.king_name = "mmm"

print(a.king_name)
print(b)