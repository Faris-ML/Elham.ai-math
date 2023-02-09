import networkx as nx
import matplotlib.pyplot as plt
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

    def backward(self,just_Variables=False):
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
                        inp.grad = inp.grad+grad
                    vis.add(inp)
        if just_Variables:
            return {self.Nodes[key].name:self.Nodes[key] for key in self.Nodes.keys() if isinstance(self.Nodes[key],Variable)}
        else:
            return {self.Nodes[key].name:self.Nodes[key] for key in self.Nodes.keys()}
    
    def update_variables(self,new_variables:dict):
        for name,node in new_variables.items():
            for key,val in self.Nodes.items():
                if key == name:
                    print('variable ', name,' updated')
                    self.Nodes[name] = val

    def plot_Graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(list(self.Nodes.keys()))
        for name,node in self.Nodes.items():
            if isinstance(node,Operator):
                for inp in [node.inp_1,node.inp_2]:
                    G.add_edge(inp.name,name)

        # separate calls to draw nodes and edges
        plt.figure(figsize=(15,8))
        pos = nx.shell_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 4500,node_color='gray',node_shape='o')
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos,arrows=True,width=2)
        plt.show()