import networkx as nx
import matplotlib.pyplot as plt

class PrintGraph:
    def __init__(self):
        G = nx.Graph()
        self.G = G
    
    def add_node(self, node, **attr):
        self.G.add_node(node, **attr)

    def add_node_list(self, data):
        for n in data:
            self.add_node(n)

    def add_edge(self, n1, n2, weight, **attr):
        self.G.add_edge(n1,n2,weight=weight, **attr)
    
    def add_edge_list(self, data):
        for i in data:
            n1, n2 = i['e']
            w = i['w']
            self.add_edge(n1, n2, w)
    
    def display(self, color_map):
        color = ['red' if node in color_map else 'green' for node in self.G]  
        pos = nx.spring_layout(self.G, seed=255) # random_layout
        plt.figure(3, figsize=(12,12)) 
        nx.draw(self.G, pos, node_color=color, with_labels=True)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)

        plt.show()
        plt.savefig("path.png")

if __name__ == '__main__':
    graph =  PrintGraph()
    node_list = [1, 2, 3, 4, 5, 6]
    edge_list = [{'e':[1, 2], 'w': 0.5}, {'e':[2, 3], 'w': 9.8}, {'e':[1, 3], 'w': 2.8}, {'e':[9, 3], 'w': 2.8}]
    graph.add_node_list(node_list)
    graph.add_edge_list(edge_list)
    graph.display()
    '''
    graph.add_node(1)
    graph.add_edge(1,2,weight=0.5)
    '''