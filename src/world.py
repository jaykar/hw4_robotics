from obstacle import Obstacle
import numpy as np
import cv2

class World(object):

    def __init__(self, obstacle_txt, goal_txt):
        self.obstacles = []
        self.goals = []
        self.populate_obstacles(obstacle_txt)
        self.populate_goal(goal_txt)

    def populate_obstacles(self, txt_file):
        with open(txt_file, 'r') as input_file:
            n_lines = int(next(input_file))
            for i in range(n_lines):
                n_vertex = int(next(input_file))
                coords = []
                for j in range(n_vertex):
                    coords.append(next(input_file).replace(' \r\n', '').split(' '))
                self.obstacles.append(Obstacle(coords))

    def populate_goal(self, goal_txt):
        f = open(goal_txt, 'r')
        coords = f.read().split('\n')
        start = map(float, coords[0].split(' '))
        end = map(float, coords[1].split(' '))
        self.goals.append(start)
        self.goals.append(end)
    
    def draw(self,grown=False, size=(600, 900, 3)):
        img = np.zeros(size)
        thickness = 2
        
        for i, obstacle in enumerate(self.obstacles):
            if i > 0:
                thickness = 1
                print 'obstacle' + str(i)
                img += obstacle.draw(grown, size,thickness, color=(0, 255, 0))
            img += obstacle.draw(False, size,thickness)
        
        cv2.imshow('world', img)
        cv2.waitKey(0)

    def _print(self):
        print "Obstacles: "  
        for obs in self.obstacles:
            print obs
        print "End goal: "
        print(self.goals)

if __name__ == '__main__':
    W = World('../input_files/hw4_world_and_obstacles_convex.txt', '../input_files/hw4_start_goal.txt')
    W.draw(grown=True)
