import sys
from minizinc import Instance, Model, Solver
sys.path.append('../')
from utils.utils import *
from datetime import timedelta

class LP_Instance():
    '''
    In this implementation, VLSI_Instance is the class representing the parsed instance to be solved (with a
    fixed width of the container and with a defined amount of rectangles to be placed inside that one)

    path: the path from which the instances are taken

    '''

    def __init__(self, path):

        rectangles = []

        with open(path, 'r') as file:
            file = file.readlines()

            for i in range(2, int(file[1]) + 2):
                width, height = file[i].split()
                rectangles.append(Rectangle(int(width), int(height)))

        self.name = path
        self.W = int(file[0])
        self.H = None
        self.n_instances = int(file[1])
        self.rectangles = rectangles

    def H_LB(self):
        '''
        In this implementation, the lower bound is computed using as best case the one in which no
        blank spaces are left

        '''

        sum = 0

        for rectangle in self.rectangles:
            sum = sum + (rectangle.height * rectangle.width)
        return int(np.ceil(sum / self.W))

    def H_UB(self, plot=False):
        '''
        In this implementation, the upper bound is computed building a first
        relaxed version of the problem: the rectangles are placed one to the right
        of another in a line starting from the bottom left corner and when the next
        rectangle is wider than the available width on the actual line, it is placed
        on a new row, over the row of the previously placed rectangles.
        H_UB is computed quickly and provides a bound which is way better than H_UB_naive.
        (this advantage is more evident with instances having many rectangles).

        '''

        W = self.W
        placer_x = 0
        occupied_height = [0] * W

        for r in self.rectangles:

            occupied_height_copy = occupied_height.copy()
            placer_x = np.argmin(occupied_height_copy)
            placer_y = min(occupied_height_copy)

            while ((placer_x + r.width) > W or any(
                    [x > placer_y for x in occupied_height[placer_x:(placer_x + r.width)]])):
                occupied_height_copy.remove(placer_y)
                placer_x = np.argmin(occupied_height_copy)
                placer_y = min(occupied_height_copy)

            # lowest_height = max(occupied_height[placer_x:(placer_x + r.width)])

            r.x = placer_x
            r.y = placer_y

            for i in range(placer_x, placer_x + r.width):
                occupied_height[i] = placer_y + r.height

            placer_x += r.width

        if plot:
            plot_rectangles(self.rectangles, title=self.name)

        return max([(r.height + r.y) for r in self.rectangles])

    def H_UB_naive(self):

        sum = 0

        for rectangle in self.rectangles:
            sum = sum + rectangle.height
        return sum

    def H_UB_rotation(self):

        sum = 0

        for rectangle in self.rectangles:
            sum = sum + min(rectangle.height, rectangle.width)
        return min(sum, self.H_UB())

    def biggest_rectangle_index(self):

        biggest_rectangle_index = 0
        area = 0
        for i in range(len(self.rectangles)):
            a = self.rectangles[i].width * self.rectangles[i].height
            if (a > area):
                area = a
                biggest_rectangle_index = i

        return biggest_rectangle_index

    def get_width_height(self):
        widths = []
        heights = []

        for rect in self.rectangles:
            widths.append(rect.width)
            heights.append(rect.height)

        return widths, heights


def lp_benchmark(start, end, timeout, verbose = True):
    timeout_ = timedelta(seconds=timeout)

    for j in range(start,end+1):
        url = f"../../instances/ins-{j}.txt"
        lp_instance = LP_Instance(url)
        width, height = lp_instance.get_width_height()

        # Load n-Queens model from file
        model = Model("./LP_alt_sb.mzn")

        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup("gurobi")

        # Create an Instance of the n-Queens model for Gecode
        mzn_instance = Instance(solver, model)

        mzn_instance["h_ub"] = lp_instance.H_UB()
        mzn_instance["h_lb"] = lp_instance.H_LB()
        mzn_instance["W"] = lp_instance.W
        mzn_instance["n_rectangles"] = lp_instance.n_instances
        mzn_instance["rect_height"] = width
        mzn_instance["rect_width"] = height
        mzn_instance["biggest_rect"] = lp_instance.biggest_rectangle_index() + 1

        # solve
        result = mzn_instance.solve(timeout_)
        print(result)
        solve_time = result.statistics["solveTime"].total_seconds()

        overtime = False

        if round(solve_time, 0) >= timeout:
            overtime = True
            if verbose:
                print(f"instance {j} overtime")
        else:
            if verbose:
                print(f"Instance {j} solved in ", solve_time)

        # plot results
        x_pos = result.solution.rect_x
        y_pos = result.solution.rect_y
        H = result.solution.lt


        for i in range(0, lp_instance.n_instances):
            lp_instance.rectangles[i].x = y_pos[i] - lp_instance.rectangles[i].width
            lp_instance.rectangles[i].y = x_pos[i] #- lp_instance.rectangles[i].height
            lp_instance.H = H
        plot_rectangles(lp_instance.rectangles, url)

    #return solve_time, overtime


lp_benchmark(1, 1, 300)

