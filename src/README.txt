README.txt
Group 7 Robotics HW4
Team Leader: Jaykar Nayeck
Arthi Suresh, Jaykar Nayeck, Joshua Dillon
as4313, jan2150, jmd2228

We implemented our code in Python 2.7.4. Before running our
program please ensure you run the following:
brew tap homebrew/science
brew install opencv
pip install numpy

To run our program, type:
python hw4.py

The input files for the world should be in a txt file in input_files called 
Press any key to quit out of the final image.

We implemented our own convex hull and dijkstra's function
but we used numpy for matrix multiplications and OpenCV to draw images.

The output of this program is as follows:
== images and GIF ==
There are four images saved:
(1) just the obstacles and start and end
(2) add grown obstacles
(3) add all possible paths
(4) highlights shortest path

We also provided a GIF that shows this process--to open
this GIF just drag it to a Chrome tab.

== text file with shortest path ==
text file that we feed into our MATLAB program that has
the shortest path and positions of the vertices

============== color key ===============
start: 						red circle
goal: 						green circle
original obstacles: 		yellow lines
grown convex hulls: 		green lines
all valid paths 
	form start to goal: 	dark blue
shortest path as given 
	by Djikstra: 			light blue
