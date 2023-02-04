from Node import *
from Graph import Graph

x=Variable(10,'x')
y=Variable(1,'y')
z=Variable(2,'z')
f=devid(y,x)
g=Graph(f)
print('befor diffrintion : ',g.forward())

print('after diffrintion : ',g.backward())