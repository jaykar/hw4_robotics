import numpy as np
import cv2
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class Obstacle(object):
    def __init__(self, xy):
        print 'new obstacle'
        self.coords = self._format(xy)
        self.coords_grown = self._grow()
        self.lines = self.get_.lines()
    
    def get_lines(self):
        lines = []
        for i in range(len(self.coords_grown)-1):
            curr_x, curr_y = self.coords_grown[i]
            next_x, next_y = self.coords_grown[i+1]
            m = (next_y - curr_y)/(next_x - curr_x)*1.0
            b = curr_y - m * curr_x
            lines.append([m, b])   
        
        curr_x, curr_y = self.coords_grown[-1]
        next_x, next_y = self.coords_grown[0]
        m = (next_y - curr_y)/(next_x - curr_x)*1.0
        b = curr_y - m * curr_x
        lines.append([m, b])
        return lines
        
    def __repr__(self):
        print self.coords
    
    def _format(self, xy):
        coords = []
        for i in range(len(xy)):
            coords.append(map(float, xy[i]))
        return coords
    
    def _grow(self, square=0.35):
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
            print pt1, pt2
            cv2.line(img, pt1, pt2, color, thickness)
        pt1, pt2 = pixel_location(coords[0], coords[-1])
        cv2.line(img, pt1, pt2, color, thickness)
        return img

    def __str__(self):
        return str(self.coords)
