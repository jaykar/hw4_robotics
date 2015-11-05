import numpy as np
import cv2

class Obstacle(object):
    def __init__(self, xy):
        self.coords = self._format(xy)

    def __repr__(self):
        print self.coords
    
    def _format(self, xy):
        coords = []
        for i in range(len(xy)):
            coords.append(map(float, xy[i]))
        return coords
    

    def draw(self, size=(600, 900, 3), thickness=1):
        def pixel_location(curr_o, next_o):
            curr_obs = np.array(curr_o) * -500/11.0 + np.array([275, 575])
            next_obs = np.array(next_o) * -500/11.0 + np.array([275, 575])
            x0, y0 = map(int, curr_obs)
            x1, y1 = map(int, next_obs)
            return ((y0, x0), (y1, x1))
        
        img = np.zeros(size)
        for i in range(len(self.coords)-1):
            pt1, pt2 = pixel_location(self.coords[i], self.coords[i+1])
            cv2.line(img, pt1, pt2, (0, 0, 255), thickness)
        pt1, pt2 = pixel_location(self.coords[0], self.coords[-1])
        cv2.line(img, pt1, pt2, (0, 0, 255), thickness)
        return img

    def __str__(self):
        return str(self.coords)
