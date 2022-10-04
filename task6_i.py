import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import random
import copy



# to create random adjacency matrix 
def random_graph2(n,m):
    adjacency_list = []
    matrix = [0]*n*n
    matrix = np.array(matrix).reshape(n,n)
    row = np.arange(0,n,1)
    all_edges = []
    for i in row :
        for j in range(i):
            edge = []
            edge.append(i)
            edge.append(j)
            all_edges.append(edge)
            

    random_edge = random.sample(all_edges,m)
    print(random_edge)

    for i in range(len(random_edge)):
        matrix[random_edge[i][0]][random_edge[i][1]] = 1
        matrix[random_edge[i][1]][random_edge[i][0]] = 1


    adjList = defaultdict(list)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
                       if matrix[i][j]== 1:
                           adjList[i].append(j)
    print(adjList)
    return adjList , matrix , random_edge

# Defining a dict
def to_dict(adjacency_list):
    
    d = defaultdict(list)
      
    for i in range(len(adjacency_list)):
        for j in range(1,len(adjacency_list[i])):
            if(adjacency_list[i][j]!=' '):
                d[i].append(int(adjacency_list[i][j]))
    return d

# random give weighted number to edge 
def random_weighted(adjacency_matrix , edge_array):
    temp = copy.deepcopy(edge_array)
    edge_weight_array = []
    for i in range(len(edge_array)):
        random_number = random.randint(1,10)
        adjacency_matrix[edge_array[i][0]][edge_array[i][1]] = adjacency_matrix\
            [edge_array[i][1]][edge_array[i][0]] = random_number

        temp[i].insert(len(temp[i]),random_number)
        edge_weight_array.append(temp[i])
    
    return adjacency_matrix,edge_weight_array

class Graph(object):

    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if v1==v2:
            print("Same vertex %d and %d" % (v1, v2))
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1

    # Remove edges
    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        # for row in self.adjMatrix:
        #     print(row)
        matrix = self.adjMatrix
        return matrix
            # for val in row:
            #     print('{:4}'.format(val)),
            # print

# to output graph 
def visualize(graph):
    G = nx.Graph()
    G.add_edges_from(graph)
    nx.draw_networkx(G)
    plt.show()

def to_array(adjacency_list):
    edge_array = []

    for i in range(len(adjacency_list)):
        for j in range(1,len(adjacency_list[i])):
            if(adjacency_list[i][j]!=' '):
                # edge_array[i].append(int([i,j]))
               
                edge_array.append([i,int(adjacency_list[i][j])])

    return edge_array


from sys import maxsize
def BellmanFord(graph, V, E, src):

	dis = [maxsize] * V

	dis[src] = 0

	for i in range(V - 1):
		for j in range(E):
            

			if (dis[graph[j][0]] + graph[j][2] < dis[graph[j][1]]):
                
				dis[graph[j][1]] = dis[graph[j][0]] + graph[j][2]
                          
	# check for negative-weight cycles.
	# The above step guarantees shortest
	# distances if graph doesn't contain
	# negative weight cycle. If we get a
	# shorter path, then there is a cycle.

	for i in range(E):
        
		x = graph[i][0]
		y = graph[i][1]
		weight = graph[i][2]
		if dis[x] != maxsize and dis[x] + \
						weight < dis[y]:
			print("Graph contains negative weight cycle")

	print("Vertex Distance from Source")
	for i in range(V):
		print("%d\t\t%d" % (i, dis[i]))
  
class Graph_struct:
  
   def __init__(self, V):
      self.V = V
      self.adj = [[] for i in range(V)]

   def DFS_Utililty(self, temp, v, visited):

      visited[v] = True

      temp.append(v)

      for i in self.adj[v]:
         if visited[i] == False:
            temp = self.DFS_Utililty(temp, i, visited)
      return temp

   def add_edge(self, v, w):
      self.adj[v].append(w)
      self.adj[w].append(v)

   def connected_components(self):
      visited = []
      conn_compnent = []
      for i in range(self.V):
         visited.append(False)
      for v in range(self.V):
         if visited[v] == False:
            temp = []
            conn_compnent.append(self.DFS_Utililty(temp, v, visited))
      return conn_compnent
class Bellman:



    def __init__(self, vertices):

        self.M = vertices   # Total number of vertices in the graph

        self.graph = []     # Array of edges



    # Add edges

    def add_edge(self, a, b, c):

        self.graph.append([a, b, c])
        # print(type(self.graph[0][0]))



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
def main():
    
    n = 100  #  nodes
    m = 500  #  edges
    # adjacency_list = random_graph(n,m)    
    print("Dictionary with values as list:")
    adjacency_list , matrix ,edge_array = random_graph2(n,m)    
    # print(adjacency_list)
    print()
   
    visualize(edge_array)
    

    g = Graph(n)    
    for i in range(m):
        g.add_edge(edge_array[i][0],edge_array[i][1])
        
    adjacency_matrix = g.print_matrix()
    
    adjancency_weight_matrix ,edge_weight_array = random_weighted(adjacency_matrix, edge_array)
    adjancency_weight_matrix_NP = np.array(adjancency_weight_matrix)
#
    
    for i in range(len(edge_weight_array)):         
            edge_weight_array[i][0] = edge_weight_array[i][0].item()

    #sort edge_array before putting into algortihm
    edge_weight_array = np.array(edge_weight_array)
    edge_weight_array.sort(axis = 1)
    edge_weight_array.sort(axis = 0)


    
    print('------------------------------------------')
    print('after dikstra algorithm :')
    # # so far , it is for creating weighted matrix 
    
    iterations_dijkstra = []
    for i in range(10):
    
            g = dikstra(n)
            g.graph = adjancency_weight_matrix
            iteration , distance = g.dijkstra(0)
            iterations_dijkstra.append(iteration)
            if(i==9):
                print()
                print('{}th iteration outcome'.format(i+1))
                print("Vertex \t Distance from Source")
                for node in range(len(distance)):
                    print(node, "\t\t", distance[node])
    
    print('------------------------------------------')
    print('average iteration of 10 times run is : {}'.format((sum(iterations_dijkstra))/len(iterations_dijkstra)))
    print()
    print()
    

    
    iterations_bellman = []
    print('------------------------------------------')
    print('after Bellman algorithm :')
    for j in range(10):
        g = Bellman(n)
        for i in range(len(edge_weight_array)):
            g.add_edge(edge_weight_array[i][0],edge_weight_array[i][1],edge_weight_array[i][2])
        iteration , distance = g.bellman_ford(0)
        iterations_bellman.append(iteration)
        if(j == 9):
            print()
            print('{}th iteration outcome'.format(j+1))
            print("Vertex Distance from Source")
            for k in range(len(distance)):    
                print("{0}\t\t{1}".format(k, distance[k]))
       
    print('------------------------------------------')

    print('average iteration of 10 times run is : {}'.format((sum(iterations_bellman))/len(iterations_bellman)))



if __name__ == '__main__':
    main()
    
    
# seed random number generators for reproducibility    
    


