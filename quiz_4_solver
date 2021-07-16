"קרדיט לאיתי חן, יואב אלינסון ולאשלי הרכבי על הקוד"
'תפריט:'
'1: שורה 149'
"שימו לב להכניס את אורך הקבל בשורה 150!"
'2: שורה 242'
'3: שורה 400'
'4: שורה 475'
"בכל פעם שמכניסים את המטריצה מהאתר ומדביקים אותה בקוד"
"-לודא שהיא נמצאת במרכאות. זה קורה בתוך הפונקציה "
"noSpace()"
import ast
import copy
from collections import defaultdict
class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.org_graph = [i[:] for i in graph]
        self.ROW = len(graph)
        self.COL = len(graph[0])
    def BFS(self, s, t, parent):

        visited = [False] * (self.ROW)

        queue = []
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:
            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    def dfs(self, graph, s, visited):
        visited[s] = True
        for i in range(len(graph)):
            if graph[s][i] > 0 and not visited[i]:
                self.dfs(graph, i, visited)

    # Returns the min-cut of the given graph
    def minCut(self, source, sink):


        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially
        while self.BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]


            max_flow += path_flow

            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        visited = len(self.graph) * [False]
        self.dfs(self.graph, s, visited)

        min_cutes_v = []
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and \
                        self.org_graph[i][j] > 0 and visited[i]:
                    print
                    str(i) + " - " + str(j)
                    min_cutes_v.append([i, j])
        return min_cutes_v

    def BFS_F(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:

                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True

        return False

    def FordFulkerson(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        while self.BFS_F(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow

def noSpace_1(str):
    str = str.replace('[ ', '[')
    str = str.replace(']', ']')
    str = str.replace(' ', ',')
    return ast.literal_eval(str)

# הכנסת שאלה 1
#ex_1
graph = Adjacemcy_matrix_G = noSpace_1("[[ 0 15 0 27 36 3 15 0 5 0] [ 0 0 33 26 22 38 24 1 16 19] [19 0 0 31 2 10 17 30 4 0] [ 0 0 0 0 12 23 10 39 0 0] [ 0 0 0 0 0 26 25 0 9 37] [ 0 0 0 0 0 0 27 25 10 22] [ 0 0 0 0 0 0 0 23 23 18] [29 0 0 0 3 0 0 0 17 6] [ 0 0 0 25 0 0 0 0 0 0] [29 0 21 28 0 0 0 0 18 0]]")
new_cable = 40
source = 0
sink = 9
graph_copy = copy.deepcopy(graph)
g = Graph(graph_copy)

min_cutes_v = g.minCut(source, sink)
print("********* ex. max flow *******")
#print("the min cut is: ", min_cutes_v)
'''for i in range(0,len(min_cutes_v)):
    index_i = min_cutes_v[i][0]
    index_j = min_cutes_v[i][1]
    print(graph[index_i][index_j])'''
graph_copy = copy.deepcopy(graph)
g = Graph(graph_copy)
#print("original max flow= ", g.FordFulkerson(source, sink))

min_max_flow = []
for i in range(0, len(min_cutes_v)):
    graph_copy = copy.deepcopy(graph)
    # cutting all the vertexes of the min cut
    for j in range(0, len(min_cutes_v)):
        index_i = min_cutes_v[j][0]
        index_j = min_cutes_v[j][1]
        graph_copy[index_i][index_j] = 0
    # replacing each time another min cut
    index_i = min_cutes_v[i][0]
    index_j = min_cutes_v[i][1]
    graph_copy[index_i][index_j] = new_cable
    g = Graph(graph_copy)
    min_max_flow.append(g.FordFulkerson(source, sink))

#print("the fixed electric matrix is:", min_max_flow)
# finding min flow
minimum = float('inf')
for i in range(0, len(min_max_flow)):
    if minimum > min_max_flow[i] and min_max_flow[i] != 0:
        minimum = min_max_flow[i]
print("the minimum max flow for G_fixed= ", minimum)
print("--------------end q 1--------------")

# QUESTION 2:
inf = float('inf')

def zeroToInf(mat):
    n = len(mat)
    for i in range(n):
        for j in range(n):
            if i != j and mat[i][j] == 0:
                mat[i][j] = inf
    return mat


def FloydWarshall(lengths):
    n = len(lengths)
    delta = [[[inf for _ in range(n)] for _ in range(n)] for _ in range(n + 1)]  # n matrices of nxn
    for i in range(n):
        if lengths[i][i] < 0:
            delta[0][i][i] = lengths[i][i]
        else:
            delta[0][i][i] = 0
    for i in range(n):
        for j in range(n):
            if lengths[i][j] != 0:
                delta[0][i][j] = lengths[i][j]

    for k in range(1, n + 1):
        for i in range(n):
            for j in range(n):
                delta[k][i][j] = min(delta[k - 1][i][j], delta[k - 1][i][k - 1] + delta[k - 1][k - 1][j])
    return delta[n][:][:]


def question2(adj):
    '''
    Input: adjacency matrix given in question
    Output: number of vertices in negative cycles
    '''
    print("\n*** QUESTION 2 ***")
    n = len(adj)
    adj = zeroToInf(adj)
    minDis = FloydWarshall(adj)
    V = []
    for i in range(n):
        if minDis[i][i] < 0:
            V.append(i)
    # V now contains vertices that are in negative cycles

    print("Number of vertices in negative cycles is: ", end="")
    print(len(V))

#שימו פה את 2
adj = noSpace_1("[[ 0 4 0 0 0 0 0 0 0 0 2 0 0 0 0] [ 0 0 0 0 0 0 0 -5 0 0 0 4 0 0 0] [ 0 0 0 0 0 5 0 -2 0 0 2 0 0 0 0] [ 0 1 0 0 0 0 0 4 0 0 0 0 0 0 0] [ 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0] [ 2 0 0 0 0 0 0 0 0 0 2 0 0 0 0] [ 0 -3 1 0 0 0 0 2 0 0 0 0 0 0 0] [ 0 0 0 0 0 0 1 0 0 0 0 0 -1 3 3] [ 0 0 5 5 1 -4 0 0 0 0 0 0 0 0 5] [ 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0] [ 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0] [ 0 0 0 0 1 0 3 0 0 -1 5 0 -3 5 0] [ 0 0 1 0 0 3 4 0 0 0 0 0 0 0 0] [ 0 0 0 0 0 3 4 0 0 2 0 0 0 0 3] [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]")
question2(adj)

print("--------------end q 2--------------")


import ast
import math
import numpy as np


def matToPerm(A):
    '''
    assuming A is a permutation matrix, returns array of length len(A) which is the permutation of rows:
    A*X causes:
    row 0 is now row S[0]
    row 1 is now row S[1]
    ...
    row n-1 is now row S[n-1]
    OUTPUT: array L
    '''
    n = len(A)
    L = [None] * n
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:  # row i was row j in I matrix
                L[j] = i
    return L


def permProduct(L1, L2):
    '''
    INPUT: 2 lists that resemble permutations: L1 and L2 which are for matrices A1,A2
    OUTPUT: list for permutation that is L1*L2
    '''
    if L2 is None:
        return None

    n = len(L1)
    ans = [None] * n
    for i in range(n):
        ans[i] = L1[L2[i]]
        # initially row i was moved to row L2[i], and now moved to L1[L2[i]]
    return ans


def add(list, tree):
    if tree.perm not in list and tree is not None:
        list.append(tree.perm)
        return tree
    else:
        return None


def enqueue(Q, node):
    if node is not None:
        Q.append(node)


class pTree:
    # children are product of self and each permutation matrix
    def __init__(self, LX, d):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.perm = LX
        self.dep = d

    def update(self, La, Lb, Lc, Ld, list):
        self.a = add(list, pTree(permProduct(La, self.perm), self.dep + 1))
        self.b = add(list, pTree(permProduct(Lb, self.perm), self.dep + 1))
        self.c = add(list, pTree(permProduct(Lc, self.perm), self.dep + 1))
        self.d = add(list, pTree(permProduct(Ld, self.perm), self.dep + 1))


def isChildEye(tree):
    boo = False
    n = len(tree.perm)
    if tree.a is not None:
        boo = boo or tree.a.perm == [*range(n)]
    if tree.b is not None:
        boo = boo or tree.b.perm == [*range(n)]
    if tree.c is not None:
        boo = boo or tree.c.perm == [*range(n)]
    if tree.d is not None:
        boo = boo or tree.d.perm == [*range(n)]
    return boo


def updateTree(root: pTree, La, Lb, Lc, Ld, list):
    n = len(root.perm)
    # using queue to update one level each time
    Q = [root]
    while Q is not None and len(Q) > 0:
        node = Q.pop(0)
        node.update(La, Lb, Lc, Ld, list)
        if isChildEye(node):
            return node.dep + 1

        enqueue(Q, node.a)
        enqueue(Q, node.b)
        enqueue(Q, node.c)
        enqueue(Q, node.d)
    return math.inf


def question3(A, B, C, D, X):
    '''
    we want to find X=A1*A2*A3*...*An (Ai are matrices A B C or D)
    => (An^-1*...*A2^-1*A1^-1)*X=I
    A,B,C,D are permutation matrices so inverse matrix is transpose
    we will check all options until:
    1. we reach I
    2. we get stuck in a loop (get to a matrix we already found)
    '''

    print("\n*** QUESTION 3 ***")
    A_t = np.transpose(A)
    B_t = np.transpose(B)
    C_t = np.transpose(C)
    D_t = np.transpose(D)

    La = matToPerm(A_t)
    Lb = matToPerm(B_t)
    Lc = matToPerm(C_t)
    Ld = matToPerm(D_t)
    Lx = matToPerm(X)

    # list of permutations we got to (as lists)
    checked = [Lx]

    root = pTree(Lx, 0)
    ans = updateTree(root, La, Lb, Lc, Ld, checked)

    print("Min number of matrices that their product is X is: ", end="")
    print(ans)


def noSpace(str):
    str = str.replace('{', '[')
    str = str.replace('}', ']')
    return ast.literal_eval(str)


'''
נרצה לכתוב את מטריצה X
'''
A = noSpace(
    "{{0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}}")
B = noSpace(
    "{{0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1,0}, {0, 0, 0, 0, 0, 0, 1}}")
C = noSpace(
    "{{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0,  0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}}")
D = noSpace(
    "{{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}}")
#הכנס כאן את 3
X = noSpace(
    "{{0, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 0, 0}}")
print("answer: ", end="")
question3(A, B, C, D, X)

print("--------------end q 3--------------")


# cities distances problem
print("\n********* ex. cities distances *******")
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph
from collections import defaultdict


class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary


    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    def KruskalMST(self, cities):
        result = []  # This will store the resultant MST
        i = 0
        e = 0
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:

            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)


            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge

        minimumCost = 0

        for u, v, weight in result:
            minimumCost += weight

        print("Minimum round-trip road", minimumCost * 2)

# 4
matrix = noSpace("{{0., 1775.07, 1827.48, 1088.53, 261.7, 1469.45, 1451.05, 496.15, 1419.16}, {1775.07, 0., 350.461, 2005.32, 1875.45, 593.561, 1387.1, 2202.76, 2886.38}, {1827.48, 350.461, 0., 1850.08, 1971.85, 902.501, 1130.25, 2292.79, 2782.86}, {1088.53, 2005.32, 1850.08, 0., 1343.39, 2025.83, 832.781, 1449.3, 976.072}, {261.7, 1875.45, 1971.85, 1343.39, 0., 1494.37, 1698.55, 327.466, 1572.59}, {1469.45, 593.561, 902.501, 2025.83, 1494.37, 0., 1638.09, 1813.92, 2772.92}, {1451.05, 1387.1, 1130.25, 832.781, 1698.55, 1638.09, 0., 1931.9, 1807.47}, {496.15, 2202.76, 2292.79, 1449.3, 327.466, 1813.92, 1931.9, 0., 1458.81}, {1419.16, 2886.38, 2782.86, 976.072, 1572.59, 2772.92, 1807.47, 1458.81, 0.}}")

cities = ["Bratislava", "Douglas", "Kyiv", "London", "Monaco", "Paris", "Podgorica", "Saint Helier", "Vilnius"]
v = 9
g = Graph(v)
# set the ne edges by the matrix
for i in range(0, v):
    for j in range(0, v):
        if matrix[i][j] != 0:
            g.addEdge(i, j, matrix[i][j])

g.KruskalMST(cities)
