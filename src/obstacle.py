import numpy as np
import cv2
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

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
        
        coords = np.array(coords) 
        hull = ConvexHull(coords)
        x = coords[hull.vertices,0]
        y = coords[hull.vertices,1]
        coords = []
        for (a, b) in zip(x,y):
            coords.append([a,b])
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
