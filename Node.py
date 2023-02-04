import math

class Node():

    def __init__(self,name):
        self.grad = None
        self.name = name
        self.value = None

class Operator(Node):

    def __init__(self,name):
        self.grad = None
        self.name = name
        self.value = None


class Variable(Node):
    def __init__(self,val,name):
        self.value = val
        self.name = name
        self.grad = None
    def forward(self):
        return self.value

class Constant(Node):
    def __init__(self,val,name):
        self.value=val
        self.name=name
    def forward(self):
        return self.value

class mul(Operator):
    def __init__(self,x1,x2,name=None):
        self.inp_1=x1
        self.inp_2=x2
        self.grad = None
        self.name = name
        self.value=None

    def forward(self):
        self.value=self.inp_1.forward() * self.inp_2.forward()
        return self.value

    def backward(self,d):
        return self.inp_2.value*d,self.inp_1.value*d


class add(Operator):
    def __init__(self, x1, x2,name=None):
        self.inp_1 = x1
        self.inp_2 = x2
        self.grad = None
        self.name=name
        self.value=None

    def forward(self):
        self.value = self.inp_1.forward() + self.inp_2.forward()
        return self.value

    def backward(self, d):
        return d, d


class sub(Operator):
    def __init__(self, x1, x2,name=None):
        self.inp_1 = x1
        self.inp_2 = x2
        self.grad = None
        self.name = name
        self.value=None

    def forward(self):
        self.value=self.inp_1.forward() - self.inp_2.forward()
        return self.value

    def backward(self, d):
        return d, -d
class power(Operator):
    def __init__(self,x1,x2,name=None):
        self.inp_1=x1
        self.inp_2=x2
        self.grad = None
        self.name = name
        self.value=None

    def forward(self):
        self.value=self.inp_1.forward()**self.inp_2.forward()
        return self.value

    def backward(self,d):
        return d*self.inp_2.value*(self.inp_1.value**(self.inp_2.value-1)), d*math.log(self.inp_1.value)*(self.inp_1.value ** self.inp_2.value)

class devid(Operator):

    def __init__(self,x1,x2,name=None):
        self.inp_1=x1
        self.inp_2=x2
        self.grad=None
        self.name = name
        self.value=None

    def forward(self):
        self.value=self.inp_1.forward()/self.inp_2.forward()
        return self.value

    def backward(self,d):
        return d/self.inp_2.value,-d*self.inp_1.value/(self.inp_2.value**2)