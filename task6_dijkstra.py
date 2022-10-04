import numpy as np
import random 
from collections import defaultdict




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

def put_weight(edges):

    for i in range(len(edges)):
        edges[i].insert(len(edges[i]),random.randint(1,10))
    return edges
def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]
    

def put_weight_to_matrix(matrix , weight_array):
    for i in range(len(weight_array)):
        matrix[weight_array[i][0]][weight_array[i][1]] = weight_array[i][2]
        matrix[weight_array[i][1]][weight_array[i][0]] = weight_array[i][2]
    return matrix 


class dikstra():
      
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = 1e7
        min_index = 5
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        iteration = 0
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
            iteration += 1
     
        # print('it ran {} times'.format(iteration)
        # self.printSolution(dist)
        return iteration , dist

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
print(edges)




matrix = [0]*node*node
matrix = np.array(matrix).reshape(node,node)

matrix_with_weight = put_weight_to_matrix(matrix,edges)
# print(matrix_with_weight)


print('------------------------------------------')
print('after dikstra algorithm :')
# # so far , it is for creating weighted matrix 

iterations_dijkstra = []
for i in range(10):

        G = dikstra(node)
        G.graph = matrix_with_weight
        iteration , distance = G.dijkstra(0)
        iterations_dijkstra.append(iteration)
        if(i==9 or i == 2):
            print()
            print('{}th iteration outcome'.format(i+1))
            print("Vertex \t Distance from Source")
            for node in range(len(distance)):
                print(node, "\t\t", distance[node])

print('------------------------------------------')
print('average iteration of 10 times run is : {}'.format((sum(iterations_dijkstra))/len(iterations_dijkstra)))
print()
print()





