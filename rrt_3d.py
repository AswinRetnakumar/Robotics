from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import random
import math
import time

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#draw cube to set graph space
r = [-5, 5]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")





class RRT_3d():
    """
    Class for RRT planning
    """

    class Node():
        """
        RRT Node
        """

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            #for drawing and path collision checking
            self.path_x = []
            self.path_y = []
            self.path_z = []
            
            self.parent = None

    def __init__(self, start, goal, obstacle_org, obs_size, rand_area,
                 expand_dis=1.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500):
        """
        Setting Parameter
        start:Start Position [x,y,z]
        goal:Goal Position [x,y,z]
        obstacleList:obstacle Positions [[centre,size],...]
        randArea:Random Sampling Area [min,max]
        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_org = obstacle_org
        self.obs_size = obs_size
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Running ")
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_org, self.obs_size):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y, self.node_list[-1].z) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_org, self.obs_size):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, step_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d, dz, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_z = [new_node.z]

        # use a maximum step distance tht the tree ca grow each step, if the node-new point 
        # distance is less than this step distance we can assign the temporary step disctance as this newly found distance
         
        if step_length > d:
            step_length = d
        # We sample the path between tree-parent and node in predefined small steps so as to check validity of path
        # Paths must not be inside the obstacles, edges must not cut the obstacle

        n_expand = math.floor(step_length / self.path_resolution)
        dz /= d
        for _ in range(int(n_expand)):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.z += dz
            new_node.path_z.append(new_node.z)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            

        d, _, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_z.append(to_node.z)


        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        return path

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.end.x
        dy = y - self.end.y
        dz = z - self.end.z
        return (dx**2 + dy**2 + dz**2)**0.5

    def get_random_node(self):
        # we can randomly sample goal as the next point so that if the goal is with in a step distance of tree we can connect tree directly to goal.
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.z)
        
        return rnd

    def draw_graph(self, rnd=None):
        
        #plt.clf()
        if rnd is not None:
            ax.scatter([rnd.x], [rnd.y], [rnd.z],"^k")

        for node in self.node_list:
            if node.parent:
                ax.plot3D(node.path_x, node.path_y, node.path_z,  "-g")

        self.plot_sphere(self.obstacle_org, self.obs_size)

        ax.scatter([self.start.x], [self.start.y], [self.start.z], "xr")
        ax.scatter([self.end.x], [self.end.y], [self.end.z], "xr")

        ax.grid(True)
        time.sleep(0.01)


    def plot_sphere(self, centre, radius):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        for c, r in zip(centre, radius):
            x = c[0]+ r*np.cos(u)*np.sin(v)
            y = c[1]+ r*np.sin(u)*np.sin(v)
            z = c[2]+ r*np.cos(v)
            ax.plot_surface(x, y, z, color="b")

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)** 2 
                +(node.z - rnd_node.z)**2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, centre, rad):

        if node is None:
            return False

        for (a, size) in zip(centre,rad):
            dx_list = [a[0] - x for x in node.path_x]
            dy_list = [a[1] - y for y in node.path_y]
            dz_list = [a[2] - z for z in node.path_z]
            d_list = [dx * dx + dy * dy + dz * dz for (dx, dy, dz) in zip(dx_list, dy_list, dz_list)]

            if min(d_list) <= (size+ 0.1)** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = (dx**2 + dy**2 + dz**2)**0.5
        theta = math.atan2(dy, dx)
        return d, dz, theta


def main(gx=1, gy=3.0, gz = 1.0):
    print("start " + __file__)
    show_animation = True
    # ====Search Path with RRT====
  # [x, y, radius]
    # Set Initial parameters
    centre = [[1.0, 1.0, 0.0], [2,-4,0.0],[-2.0, -2.0, -2.0], [-2.0, 0.0, 2.0]]
    radius = [2.0, 1.5, 0.8, 1.5]
    rrt = RRT_3d(start= [-4.0, -4.0, -4.0],
              goal= [gx, gy, gz],
              rand_area=[-5, 5],
              obstacle_org=centre,
              obs_size = radius)
    path = rrt.planning(animation=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print("Path", path)
        # Draw final path
        if show_animation:
            rrt.draw_graph()
            ax.plot3D([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-r')
            ax.grid(True)
            plt.show()


if __name__ == '__main__':
    main()
