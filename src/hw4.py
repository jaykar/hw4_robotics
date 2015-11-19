# -*- coding: utf-8 -*-
import numpy as np
import cv2
from math import sqrt
import numpy as np
import cv2
import math

def find_convex_hull(pts_array):
    num_pts = len(pts_array);
    
    # find the rightmost, lowest point, label it P0
    sorted_pts = sorted(pts_array, key=lambda element: (element[0], -element[1]))
    P0 = sorted_pts.pop()
    P0_x = P0[0]
    P0_y = P0[1]
    
    # sort all points angularly about P0
    # Break ties in favor of closeness to P0
    # label the sorted points P1....PN-1
    sort_array = [];
    for i in range(num_pts-1):
        x = pts_array[i][0]
        y = pts_array[i][1]
        angle = 0
        x_diff = x - P0_x
        y_diff = y - P0_y
        angle = math.degrees(math.atan2(y_diff, x_diff))
        if angle < 0:
            angle = angle * (-1) + 180
        dist = math.degrees(math.sqrt(x_diff**2 + y_diff**2));
        pt_info = (round(angle, 3), round(dist,3), pts_array[i])
        sort_array.append(pt_info)
    sorted_pts = sorted(sort_array, key=lambda element: (element[0], element[1]))
    # Push the points labeled PN−1 and P0 onto a stack. T
    # these points are guaranteed to be on the Convex Hull
    pt_stack = []
    pt_stack.append(sorted_pts[num_pts - 2][2])
    pt_stack.append(P0)
    
    # Set i = 1
    # While i < N do
    # If Pi is strictly left of the line formed by top 2 stack entries (Ptop, Ptop−1), 
    # then Push Pi onto the stack and increment i; else Pop the stack (remove Ptop).    
    i = 1
    while i < num_pts - 1:
        P_i = sorted_pts[i][2]
        c = pt_stack.pop()
        d = pt_stack.pop()
        pt_stack.append(d)
        pt_stack.append(c)
        # find the line formed by these two points and see if the point Pi is
        # strictly to the left of this line
        is_to_the_left = False
        position = (d[0] - c[0]) * (P_i[1] - c[1]) - (d[1] - c[1]) * (P_i[0] - c[0]) 
        if position < 0:
            is_to_the_left = True

        if (is_to_the_left):
            pt_stack.append(P_i)
            i += 1;
        else:
            pt_stack.pop()
    return pt_stack[:-1]
# #pts = [[0.0, 1.0], [1.0, 5.0], [2.0, 3.0], [2.0, 3.0], [3.0, 5.0], [3.0, 2.0], [4.0, 2.0], [6.0, 3.0]];
# #pts = [[-0.111, -1.374], [-0.111, -2.174], [-0.911, -1.374], [-0.911, -2.174], [-0.111, -1.843], [-0.111, -2.643], [-0.911, -1.843], [-0.911, -2.643], [0.699, -1.843], [0.699, -2.643], [-0.101, -1.843], [-0.101, -2.643], [0.699, -1.374], [0.699, -2.174], [-0.101, -1.374], [-0.101, -2.174]]                    
# pts = [[1.588, -1.373], [1.588, -2.173], [0.788, -1.373], [0.788, -2.173], [1.588, -1.843], [1.588, -2.643], [0.788, -1.843], [0.788, -2.643], [1.118, -1.843], [1.118, -2.643], [0.318, -1.843], [0.318, -2.643], [1.118, -1.373], [1.118, -2.173], [0.318, -1.373], [0.318, -2.173]]
# hull = find_convex_hull(pts);      
# print(hull)       
         

class Line(object):
    def __init__(self, start, end):
        self.x0 = start[0]
        self.y0 = start[1]
        self.x1 = end[0]
        self.y1 = end[1]
        self.constant, self.m  = self.gradient()
        self.b  = self.y_intercept()

        self.A = self.y1 - self.y0
        self.B = self.x0 - self.x1
        self.C = self.A*self.x0 + self.B*self.y0

    def line_intersection(self,line):
        det = self.A * line.B - line.A * self.B
        xpts = [line.x0, line.x1]
        ypts = [line.y0, line.y1]
        min_x, min_y = min(xpts), min(ypts)
        max_x, max_y = max(xpts), max(ypts)

        min_x = round(min_x, 3)
        max_x = round(max_x, 3)
        min_y = round(min_y, 3)
        max_y = round(max_y, 3)


        xpts = [self.x0, self.x1]
        ypts = [self.y0, self.y1]
        min_x_p, min_y_p = min(xpts), min(ypts)
        max_x_p, max_y_p = max(xpts), max(ypts)

        min_pt_x = round(min_x_p, 3)
        max_pt_x = round(max_x_p, 3)
        min_pt_y = round(min_y_p, 3)
        max_pt_y = round(max_y_p, 3)

        if det == 0:
            return False
        else:
            x = round((line.B * self.C - self.B * line.C)/det, 3)
            y = round((self.A * line.C - line.A * self.C)/det, 3)
            
            #print max_y - y
            #print "intersection x, y", x, y
            #print "self", (self.x0,  self.y0), (self.x1, self.y1)
            #print "line", (line.x0,  line.y0), (line.x1, line.y1)
            #print "min_x, max_x", min_x, max_x
            #print "min_y, max_y", min_y, max_y
            a = (x > min_x and x < max_x)
            b = (y > min_y and y < max_y)
            c = (x <= max_pt_x and x >= min_pt_x)
            d = (y <= max_pt_y and y >= min_pt_y)
            #print "conditional", a, b
            if (a or b) and (c and d):
               return True
            else:
                return False

    def gradient(self):
        y_p = self.y1 - self.y0 
        x_p = self.x1 - self.x0 
        if x_p == 0:
            return self.y1, 'inf'
        elif y_p == 0:
            return self.x1, 0
        else:
            return None, y_p/x_p

    def y_intercept(self):
        if self.constant:
            if self.m == 'inf':
                return None
            if self.m == 0:
                return self.x1
        else:
            return self.y1 - self.m * self.x1

    def compare(self, line):
        return (self.m == line.m and self.b == line.b and \
            self.constant == line.constant)

    def intersect(self, all_lines):
        for line in all_lines:
            intersect_in_range = self.line_intersection(line)
            if intersect_in_range:
                return True
        return False

    def __str__(self):
        if self.constant:
            if self.m == 'inf':
                return "x = " + str(self.constant)
            else:
                return "y = " + str(self.constant)
        else:
            return "y = " + str(self.m) + "x + " + str(self.b)

class Obstacle(object):
    def __init__(self, xy, grow=True):
        self.coords = self._format(xy)
        if grow:
            self.coords_grown = self._grow()
            self.lines = self.obstacle_lines(self.coords_grown)
            self.lines.extend(self.obstacle_lines(self.coords))
        else:
            self.coords_grown = self.coords[:]
            self.lines = self.obstacle_lines(self.coords_grown)        

    def obstacle_lines(self, A):
        lines = []
        for i in range(len(A)-1):
            curr_x, curr_y = A[i]
            next_x, next_y = A[i+1]
            lines.append(Line([curr_x, curr_y], [next_x, next_y]))
        curr_x, curr_y = A[-1]
        next_x, next_y = A[0]
        lines.append(Line([curr_x, curr_y], [next_x, next_y]))
        return lines
        
    def __repr__(self):
        print self.coords
    
    def _format(self, xy):
        coords = []
        for i in range(len(xy)):
            coords.append(map(float, xy[i]))
        return coords
    
    def _grow(self, square=0.4):
        coords = []
        for x,y in self.coords:
            coords.append([x+square, y+square])
            coords.append([x+square, y-square])
            coords.append([x-square, y+square])
            coords.append([x-square, y-square])
        
        coords_round = []
        for coord in coords:
            x, y = coord
            coords_round.append([round(x, 3), round(y,3)])
        
        coords = coords_round
        #print "obstacle", coords

        coords = find_convex_hull(coords_round)
        
        #coords = np.array(coords) 
        #hull = ConvexHull(coords)
        #x = coords[hull.vertices,0]
        # y = coords[hull.vertices,1]
        # coords = []

        # for (a, b) in zip(x,y):
        #    coords.append([a,b])
        return coords
    
    def draw(self, grown=False, size=(600, 900, 3), thickness=1, color =(0, 0, 255)):
        def pixel_location(curr_o, next_o):
            curr_obs = np.array(curr_o) * -500/11.0 + np.array([275, 575])
            next_obs = np.array(next_o) * -500/11.0 + np.array([275, 575])
            x0, y0 = map(int, curr_obs)
            x1, y1 = map(int, next_obs)
            return ((y0, x0), (y1, x1))
        
        coords = [] 
        if grown: 
            coords = self.coords_grown
        else:
            coords = self.coords
        
        img = np.zeros(size)
        for i in range(len(coords)-1):
            pt1, pt2 = pixel_location(coords[i], coords[i+1])
            cv2.line(img, pt1, pt2, color, thickness)
        pt1, pt2 = pixel_location(coords[0], coords[-1])
        cv2.line(img, pt1, pt2, color, thickness)
        return img

    def __str__(self):
        return str(self.coords)





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
        ans = self.djikstra(self.nodes, s, e)
        if len(ans) != 0:
            return ans
        else:
            return "no path"

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

        cv2.imwrite("map.png", img)
        #cv2.imshow('world', img)
        #cv2.waitKey(0)

        for i, obstacle in enumerate(self.obstacles):
            if i > 0:
                thickness = 1
                img += obstacle.draw(True, size,thickness, color=(0, 255, 0))
            img += obstacle.draw(False, size,thickness)

        
        cv2.imwrite("map_grown_obstacle.png", img)

        for key in self.graph.nodes:
           curr = self.info[key]
           start = pixel_location(curr)
           for key2 in self.graph.nodes[key]:
               next = self.info[key2]
               end = pixel_location(next)
               cv2.line(img, start, end, color=(255,0,0), thickness=1)
        #cv2.imshow('world', img)
        #cv2.waitKey(0)
        cv2.imwrite("map_all_path.png", img)

        path = self.path
        for i in range(len(path) -1):
            key, key2 = path[i], path[i+1]
            curr = self.info[key]
            start = pixel_location(curr)
            next = self.info[key2]
            end = pixel_location(next)
            cv2.line(img, start, end, color=(255,255,0), thickness=1)

        cv2.imwrite("map_valid_path.png", img)
        cv2.imshow('world', img)
        cv2.waitKey(0)

    def output_path(self, output_file):
        f = open(output_file, 'w')
        for node in self.path:
            coord = self.info[node]
            x, y = coord
            #position = "X pos: " + str(x) + " Y pos: " + str(y) + "\n"
            f.write(str(x) + ' '+ str(y) + '\n')
        f.close()

if __name__ == '__main__':
    W = World('hw4_world_and_obstacles_convex.txt', 'hw4_start_goal.txt')
    #W = World('../input_files/small_world.txt', '../input_files/hw4_start_goal.txt')
    W.draw(grown=True)
    W.output_path('output.txt')
    print W.path
