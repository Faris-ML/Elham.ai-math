from Node import *
from Graph import Graph
import numpy as np

x=Constant(np.array([1,2,3]),'x')
y=Constant(np.array([2,3,4]),'z')
w=Variable(8,'w')
f=power(x,y)
g=Graph(f)
print('forward : ',g.forward())

print('backward : ',g.backward())