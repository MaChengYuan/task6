import numpy as np
import random 
from collections import defaultdict
class Bellman:

    def __init__(self, vertices):

        self.M = vertices   # Total number of vertices in the graph

        self.graph = []     # Array of edges

    # Add edges

    def add_edge(self, a, b, c):

        self.graph.append([a, b, c])



    # Print the solution

    def print_solution(self, distance):

        print("Vertex Distance from Source")

        for k in range(self.M):

            print("{0}\t\t{1}".format(k, distance[k]))



    def bellman_ford(self, src):
      
        distance = [float("Inf")] * self.M

        distance[src] = 0

        iteration = 0 


        for _ in range(self.M - 1):

            for a, b, c in self.graph:
                iteration += 1

                if distance[a] != float("Inf") and distance[a] + c < distance[b]:

                    distance[b] = distance[a] + c


        for a, b, c in self.graph:

            if distance[a] != float("Inf") and distance[a] + c < distance[b]:
                print("Graph contains negative weight cycle")

                return      
        # self.print_solution(distance)
        return iteration , distance


# create edges 
def create_random_edge(column_num , edge_num):
    edges = []
    node = np.arange(0,column_num,1)
    for i in node :
        for j in range(i):
            edge= []
            edge.append(i)
            edge.append(j)
            edges.append(edge)
     
    random_edge = random.sample(edges,edge_num)
    return random_edge

#add weight to edges

def put_weight(edges):

    for i in range(len(edges)):
        edges[i].insert(len(edges[i]),random.randint(1,10))
    return edges
def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]
    
#create adjacency matrix with weight
def put_weight_to_matrix(matrix , weight_array):
    for i in range(len(weight_array)):
        matrix[weight_array[i][0]][weight_array[i][1]] = weight_array[i][2]
        matrix[weight_array[i][1]][weight_array[i][0]] = weight_array[i][2]
    return matrix 




node = 100
edge = 200

edges = create_random_edge(node,edge)
edges =np.array(edges)
Swap(edges,1,0)
edges = edges.tolist()
print(edges)
edges.sort(key=lambda i: i[1])
edges.sort(key=lambda i: i[0])
edges = put_weight(edges)
edges =np.array(edges)
print(edges)





matrix = [0]*node*node
matrix = np.array(matrix).reshape(node,node)

matrix_with_weight = put_weight_to_matrix(matrix,edges)



iterations_bellman = []
print('------------------------------------------')
print('after Bellman algorithm :')
for j in range(10):
    g = Bellman(node)
    for i in range(len(edges)):
        g.add_edge(edges[i][0],edges[i][1],edges[i][2])
    iteration , distance = g.bellman_ford(0)
    iterations_bellman.append(iteration)
    if(j == 9 or j == 2 or j == 5):
        print()
        print('{}th iteration outcome'.format(j+1))
        print("Vertex Distance from Source")
        for k in range(len(distance)):    
            print("{0}\t\t{1}".format(k, distance[k]))
   
print('------------------------------------------')

print('average iteration of 10 times run is : {}'.format((sum(iterations_bellman))/len(iterations_bellman)))





