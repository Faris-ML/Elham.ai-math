from numpy import isin
from Node import *
class Graph():
    Nodes={}
    count=0
    def __init__(self,root:Node):
        self.root=root
        self.sort()

    def sort(self):
        vis=set()
        def check(self,node:Node):
            if node not in vis:
                vis.add(node)
                if isinstance(node, Operator):
                    for input_node in [node.inp_1,node.inp_2]:
                        check(self,input_node)
                    if node.name == None:
                        node.name='Operator'+str(self.count)
                        self.count = self.count + 1
                    self.Nodes[node.name]=node
                elif isinstance(node, Variable):
                    self.Nodes[node.name] = node
                elif isinstance(node, Constant):
                    self.Nodes[node.name] = node
        check(self,self.root)

    def forward(self):
        return self.root.forward()

    def backward(self,just_Variabls=False):
        vis=set()
        self.Nodes[list(self.Nodes.keys())[-1]].grad=1
        for key in reversed(list(self.Nodes.keys())):
            if isinstance(self.Nodes[key], Operator):
                inputs=[self.Nodes[key].inp_1,self.Nodes[key].inp_2]
                grads=self.Nodes[key].backward(d=self.Nodes[key].grad)
                for inp, grad in zip(inputs, grads):
                    if inp not in vis:
                        inp.grad = grad
                    else:
                        inp.grad += grad
                    vis.add(inp)
        if just_Variabls:
            return {self.Nodes[key].name:self.Nodes[key].grad for key in self.Nodes.keys() if isinstance(self.Nodes[key],Variable)}
        else:
            return {self.Nodes[key].name:self.Nodes[key].grad for key in self.Nodes.keys()}
    
    def update_variables(self,new_variables:dict):
        for name,node in self.Nodes:
            if isinstance(node,Variable):
                for key,val in new_variables.items():
                    if key == name:
                        self.Nodes[name]=new_variables[key]