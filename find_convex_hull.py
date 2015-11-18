import math

def find_convex_hull(pts_array):
    num_pts = len(pts_array);
    
    # find the rightmost, lowest point, label it P0
    sorted_pts = sorted(pts_array, key=lambda element: (element[0], -element[1]))
    P0 = sorted_pts.pop()
    print(P0)
    P0_x = P0[0];
    P0_y = P0[1];
    
    # sort all points angularly about P0
    # Break ties in favor of closeness to P0
    # label the sorted points P1....PN-1
    sort_array = [];
    for i in range(num_pts-1):
        x = pts_array[i][0]
        y = pts_array[i][1]
        angle = 0;
        x_diff = x - P0_x
        y_diff = y - P0_y
        angle = math.degrees(math.atan2(y_diff, x_diff));
        if angle < 0:
            angle = angle * (-1) + 180
        dist = math.degrees(math.sqrt(x_diff**2 + y_diff**2));
        pt_info = (round(angle, 3), round(dist,3), pts_array[i])
        sort_array.append(pt_info)
    sorted_pts = sorted(sort_array, key=lambda element: (element[0], element[1]))
    print(sorted_pts)
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
    while i < num_pts - 2:
        P_i = sorted_pts[i][2];
        c = pt_stack.pop();
        d = pt_stack.pop();
        print(P_i, c, d)
        pt_stack.append(d);
        pt_stack.append(c);
        # find the line formed by these two points and see if the point Pi is
        # strictly to the left of this line
        is_to_the_left = False
        if c[0] != d[0]: # not a vertical line in conventional xy plane
            m = (c[1] - d[1])/(c[0] - d[0])
            b = c[1] - m*(c[0])
            print(m, b)
            if m == 0: 
                if (P_i[0] < min(c[0], d[0])):
                    is_to_the_left = True
            else:
                x_line = (P_i[1] - b)/m
                if P_i[0] < x_line:
                    is_to_the_left = True
        else: 
            if P_i[0] < c[0]:
                is_to_the_left = True
        print(is_to_the_left)
        if (is_to_the_left):
            pt_stack.append(P_i);
            i += 1;
        else:
            pt_stack.pop();
        print("revised stack:")
        print(pt_stack)
        print('\n')
    return pt_stack;

pts = [[0, 1], [1, 5], [2, 3], [2, 3], [3, 5], [3, 2], [4, 2], [6, 3]];                      
hull = find_convex_hull(pts);      
print(hull)       
         