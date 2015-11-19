from obstacle import Obstacle
from obstacle import Line
import numpy as np
import cv2
from math import sqrt


class Graph():
    def __init__(self):
        self.nodes = {}

    def node_exists(self, a):
        return a in self.nodes

    def add_node(self, a):
        if not self.node_exists(a):
            self.nodes[a] = {}

    def add_edge(self, a, b, weight):
        if not self.node_exists(a):
            self.add_node(a)
        if not self.node_exists(b):
            self.add_node(b)
        self.nodes[a][b] = weight
        self.nodes[b][a] = weight

    def shortest_path(self):
        s = 'start'
        e = 'end'
        return self.djikstra(self.nodes, s, e)

    def djikstra(self, nodes, S, G):
        #nodes = graph.nodes
        unvisited = set(nodes.keys())
        unvisited.remove(S)

        # initialize distance dictionary    
        dist = {}
        predecessors = {}
        dist[S] = 0;
        predecessors[S] = S
        S_neighbors = set(nodes[S].keys());
        for n in unvisited:
            if n in S_neighbors:
                dist[n] = nodes[S][n]
                predecessors[n] = S
            else:
                dist[n] = float('Inf');
      
        while len(unvisited) > 0:
            # find the closest unvisited node        
            keys = unvisited.intersection(set(dist.keys()))
            dist_of_unvisited = {k:dist[k] for k in keys}
            V = min(dist_of_unvisited, key = dist_of_unvisited.get);
            unvisited.remove(V);
            V_neighbors = set(nodes[V].keys())
            for W in V_neighbors:
                if ((dist[V] + nodes[V][W]) < dist[W]):
                    dist[W] = dist[V] + nodes[V][W]
                    predecessors[W] = V           
        path = []
        end = G;
        while end != S:
            path.append(end)
            end = predecessors[end]
        path.reverse()
        return path

class World(object):

    def __init__(self, obstacle_txt, goal_txt):
        self.obstacles = []
        self.goals = []
        self.populate_obstacles(obstacle_txt)
        self.populate_goal(goal_txt)
        self.info, self.graph, self.all_lines = self.intialize_graph()
        self.make_edges()
        self.path = ['start']
        self.path.extend(self.graph.shortest_path())
        self.get_matlab_instructions('1')
    def position_nodes(self, coord):
        x, y = coord
        a = np.eye(4)
        theta = np.arctan2(y, x)
        rot = np.array([ [np.cos(theta) , np.sin(theta), 0.0] ,\
                         [-np.sin(theta), np.cos(theta), 0.0] ,\
                         [0.0           , 0.0          , 1.0] ,\
                         ])
        a[:3, :3] = rot
        a[0, 3]   = x
        a[1, 3]   = y
        return a

    def get_matlab_instructions(self, text_file):
        # x,y = self.info[self.path[0]]
        # prev = np.eye(4)
        # prev[0, 3]   = x
        # prev[1, 3]   = y
        prev_angle = 0
        for i in range(len(self.path) - 1):
            curr = self.info[ self.path[i] ]
            x0, y0 = curr
            next = self.info[ self.path[i+1] ]
            x1, y1 = next
            theta = np.arctan2(y1 - y0, x1 - x0) #- prev_angle
            prev_angle = theta
            print "rotate", 90 - (theta - (np.pi/2))*180/np.pi
            print "move", np.sqrt((y1 - y0)**2 + (x1 - x0)**2 )

    def intialize_graph(self):
        count = 0
        graph = Graph()
        info = {}
        lines = []
        graph.add_node('start')
        info['start'] = self.goals[0]

        for j,obs in enumerate(self.obstacles[1:]):
        #for j,obs in enumerate(self.obstacles):
            for i in range(len(obs.coords_grown)):
                key = str(j) + "_" +str(i)
                graph.add_node(key)
                info[key] = obs.coords_grown[i]
            lines.extend(obs.lines)
        graph.add_node('end')
        info['end'] = self.goals[1]
        lines.extend(self.obstacles[0].lines)
        return info, graph, lines

    def make_edges(self):
        nodes = list(self.graph.nodes.keys())
        n_nodes = len(nodes)

        for i in range(n_nodes - 1):
            curr_node = nodes[i]
            for j in range(i + 1, n_nodes):
                next_node = nodes[j]
                #print str(curr_node) + " -> " + str(next_node)
                curr_line = Line(self.info[curr_node], self.info[next_node])
                obs_n = next_node[0]
                obs_l = curr_node[0]

                ret = curr_line.intersect(self.all_lines)
                #print ret
                #print ""
                if not ret:
                    x0, y0 = self.info[curr_node]
                    x1, y1 = self.info[next_node]
                    weight = np.sqrt( (x0-x1)**2 + (y0 - y1)**2 )
                    self.graph.add_edge(curr_node, next_node, weight)
    
    def populate_obstacles(self, txt_file):
        with open(txt_file, 'r') as input_file:
            n_lines = int(next(input_file))
            for i in range(n_lines):
                n_vertex = int(next(input_file))
                coords = []
                for j in range(n_vertex):
                    coords.append(next(input_file).replace(' \r\n', '').split(' '))
                if i==0:
                    self.obstacles.append(Obstacle(coords, grow=False))
                else:
                    self.obstacles.append(Obstacle(coords, grow=True))

    def populate_goal(self, goal_txt):
        f = open(goal_txt, 'r')
        coords = f.read().split('\n')
        start = map(float, coords[0].split(' '))
        end = map(float, coords[1].split(' '))
        self.goals.append(start)
        self.goals.append(end)
    
    def draw(self,grown=False, size=(600, 900, 3)):
        def pixel_location(curr_o):
            curr_obs = np.array(curr_o) * -500/11.0 + np.array([275, 575])
            x0, y0 = map(int, curr_obs)
            return (y0, x0)
        
        img = np.zeros(size)
        thickness = 2
        
        for i, obstacle in enumerate(self.obstacles):
            if i > 0:
                thickness = 1
                img += obstacle.draw(False, size,thickness, color=(0, 255, 0))
            img += obstacle.draw(False, size,thickness)

        start =  pixel_location(self.goals[0])
        end =  pixel_location(self.goals[1])
        cv2.circle(img, start, radius = 5, color=(0,0,255)) 
        cv2.circle(img, end, radius = 5, color=(0,255,0)) 
        #cv2.imshow('world', img)
        #cv2.waitKey(0)

        for i, obstacle in enumerate(self.obstacles):
            if i > 0:
                thickness = 1
                img += obstacle.draw(True, size,thickness, color=(0, 255, 0))
            img += obstacle.draw(False, size,thickness)

        img2 = np.copy(img)
        cv2.imshow('world', img)
        cv2.waitKey(0)

        for key in self.graph.nodes:
           curr = self.info[key]
           start = pixel_location(curr)
           for key2 in self.graph.nodes[key]:
               next = self.info[key2]
               end = pixel_location(next)
               cv2.line(img, start, end, color=(255,0,0), thickness=1)
        cv2.imshow('world', img)
        cv2.waitKey(0)

        path = self.path
        for i in range(len(path) -1):
            key, key2 = path[i], path[i+1]
            curr = self.info[key]
            start = pixel_location(curr)
            next = self.info[key2]
            end = pixel_location(next)
            cv2.line(img2, start, end, color=(255,0,0), thickness=1)

        cv2.imshow('world', img2)
        cv2.waitKey(0)

    def _print(self):
        print "Obstacles: "  
        for obs in self.obstacles:
            print obs
        print "End goal: "
        print(self.goals)

if __name__ == '__main__':
    W = World('../input_files/hw4_world_and_obstacles_convex.txt', '../input_files/hw4_start_goal.txt')
    #W = World('../input_files/small_world.txt', '../input_files/hw4_start_goal.txt')
    W.draw(grown=True)
    print W.path
