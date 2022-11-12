
import sys

import minizinc
from minizinc import Instance, Model, Solver

sys.path.append('../')
from utils.utils import *
from datetime import timedelta
from tqdm import tqdm

class LP_Instance:
    """
    In this implementation, VLSI_Instance is the class representing the parsed instance to be solved (with a
    fixed width of the container and with a defined amount of rectangles to be placed inside that one)

    path: the path from which the instances are taken

    """

    def __init__(self, path):

        rectangles = []

        with open(path, 'r') as file:
            file = file.readlines()

            for j in range(2, int(file[1]) + 2):
                width, height = file[j].split()
                rectangles.append(Rectangle(int(width), int(height)))

        self.name = path
        self.W = int(file[0])
        self.H = None
        self.n_instances = int(file[1])
        self.rectangles = rectangles

        for j in range(0, len(self.rectangles)):
            if self.rectangles[j].width > self.W:
                raise Exception(f"The width of the rectangle n.{j} is over the container width W = {self.W}")

    def H_LB(self):
        """
        In this implementation, the lower bound is computed using as best case the one in which no
        blank spaces are left

        """

        height = 0

        for rectangle in self.rectangles:
            height += (rectangle.height * rectangle.width)
        return int(np.ceil(height / self.W))

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

        height = 0

        for rectangle in self.rectangles:
            height += rectangle.height
        return height

    def H_UB_rotation(self):

        height = 0

        for rectangle in self.rectangles:
            height += min(rectangle.height, rectangle.width)
        return min(height, self.H_UB())

    def biggest_rectangle_index(self):

        biggest_rectangle_index = 0
        smaller_rectangles = []

        area = 0
        for j in range(len(self.rectangles)):
            a = self.rectangles[j].width * self.rectangles[j].height
            if a > area:
                area = a
                biggest_rectangle_index = j + 1

        for j in range(len(self.rectangles)):
            if self.rectangles[j].width > (self.W - area) // 2:
                smaller_rectangles.append(j + 1)

        return biggest_rectangle_index, smaller_rectangles

    def get_width_height(self):
        widths = []
        heights = []

        for rect in self.rectangles:
            widths.append(rect.width)
            heights.append(rect.height)

        return widths, heights

    def get_large_rectangles_index(self):
        large_rectangles = []
        for i in range(0, len(self.rectangles) - 1):
            for j in range(i + 1, len(self.rectangles)):
                if self.rectangles[i].width + self.rectangles[j].width > self.W:
                    large_rectangles.append([i + 1, j + 1])

        return large_rectangles

    def get_same_dim_rectangles_index(self):
        same_dim_rectangles = []
        for i in range(0, len(self.rectangles) - 1):
            for j in range(i + 1, len(self.rectangles)):
                if self.rectangles[i].width == self.rectangles[j].width and self.rectangles[i].height == \
                        self.rectangles[j].height:
                    same_dim_rectangles.append([i + 1, j + 1])

        return same_dim_rectangles

    def get_squares_index(self):
        squares = []
        for i in range(0, len(self.rectangles)):
            if self.rectangles[i].is_square():
                squares.append(i + 1)
        return squares


def cp_benchmark(index, timeout, method, solver_name, verbose=True, plot=True):
    """
    performs a benchmark in which instances having index from
    start to end are tested
    """

    timeout_ = timedelta(seconds=timeout)

    folder = "../instances/"
    file = f"ins-{index}.txt"
    url = folder + file
    lp_instance = LP_Instance(url)
    width, height = lp_instance.get_width_height()

    # Load 2d-strip-packaging model from file
    if method == "rotations":
        model = Model("./CP_rotated.mzn")
    elif method == "rotations-sb":
        model = Model("./CP_rotated_SB.mzn")
    elif method == "base-sb":
        model = Model("./CP_sb.mzn")
    elif method == "base":
        model = Model("./CP.mzn")
    else:
        model = Model("./CP.mzn")

    # Find the MiniZinc solver configuration
    solver = Solver.lookup(solver_name)

    # Create an Instance of the 2d-strip-packaging model
    mzn_instance = Instance(solver, model)

    if method == "rotations-sb":
        squares = lp_instance.get_squares_index()
        mzn_instance["n_squares"] = len(squares)
        mzn_instance["square_index"] = squares

    if method == "base-sb":
        biggest_rectangle, _ = lp_instance.biggest_rectangle_index()
        mzn_instance["biggest_rect"] = biggest_rectangle

    mzn_instance["h_ub"] = lp_instance.H_UB()
    mzn_instance["fixed_width"] = lp_instance.W
    mzn_instance["n_components"] = lp_instance.n_instances
    mzn_instance["heights"] = height
    mzn_instance["widths"] = width

    try:
        # solve
        result = mzn_instance.solve(timeout=timeout_, random_seed=7, free_search=True)

        if verbose:
            print(result)
        solve_time = result.statistics["solveTime"].total_seconds()

        time_over = False

        if round(solve_time + 0.6, 0) >= timeout or (result.solution is None):
            time_over = True
            solve_time = 301
            if verbose:
                print(f"instance {index} overtime")
        else:
            if verbose:
                print(f"Instance {index} solved in ", solve_time)

            if plot:
                # plot results
                x_pos = result.solution.x_coords
                y_pos = result.solution.y_coords
                H = result.solution.height

                if method == "rotations" or method == "rotations-sb":
                    rotated = result.solution.rot
                    for i in range(0, lp_instance.n_instances):
                        if rotated[i] == 1:
                            new_width = lp_instance.rectangles[i].height
                            new_height = lp_instance.rectangles[i].width
                            lp_instance.rectangles[i].width = new_width
                            lp_instance.rectangles[i].height = new_height

                for i in range(0, lp_instance.n_instances):
                    lp_instance.rectangles[i].x = x_pos[i]
                    lp_instance.rectangles[i].y = y_pos[i]
                lp_instance.H = H
                plot_rectangles(lp_instance.rectangles, url)
    except minizinc.error.MiniZincError:
        print(f"Instance {index} not solved")
        for i in range(0, lp_instance.n_instances):
            lp_instance.rectangles[i].x = None
            lp_instance.rectangles[i].y = None
        lp_instance.H = None
        solve_time = 301
        time_over = True

    path = method + "/" + solver_name
    write_log(path="../out/cp/" + path + "/" + file, instance=lp_instance,
              add_text="\n" + str(solve_time) + "\n" + str(time_over))
    return solve_time, time_over


def plot_CP_benchmark(instances_to_solve, solver_name, timeout=300, plot=False):
    """
    Produces the barplot with all the LP solving mechanisms (base, rotations,
    base + symmetry breaking, rotations + symmetry breaking). Also produces a
    file with initial values + final positioning coordinates and total height.
    In the same folder an output with the resulting time statistics is produced.

    """

    times_base = []
    times_base_rotate = []
    times_SB = []
    times_SB_rotate = []

    time_overs_base = []
    time_overs_base_rotate = []
    time_overs_SB = []
    time_overs_SB_rotate = []

    for j in tqdm(range(1, instances_to_solve + 1)):

        # base
        time, time_over = cp_benchmark(j, timeout=timeout, method="base", solver_name=solver_name, plot=plot)
        times_base.append(time)

        if time_over:
            time_overs_base.append(j - 1)

        # base + SB
        time, time_over = cp_benchmark(j, timeout=timeout, method="base-sb", solver_name=solver_name, plot=plot)
        times_SB.append(time)

        if time_over:
            time_overs_SB.append(j - 1)

        # rotated
        time, time_over = cp_benchmark(j, timeout=timeout, method="rotations", solver_name=solver_name, plot=plot)
        times_base_rotate.append(time)

        if time_over:
            time_overs_base_rotate.append(j - 1)

        # rotated + SB
        time, time_over = cp_benchmark(j, timeout=timeout, method="rotations-sb", solver_name=solver_name, plot=plot)
        times_SB_rotate.append(time)

        if time_over:
            time_overs_SB_rotate.append(j - 1)

    X = range(1, instances_to_solve + 1)
    X_axis = np.arange(0, len(times_SB) * 2, 2)

    plt.rcParams["figure.figsize"] = (13, 6)
    plt.xticks(X_axis, X)

    # base
    barbase = plt.bar(X_axis - 0.6, times_base, 0.4, label='Base time')

    for j in time_overs_base:
        barbase[j].set_alpha(0.25)

    # base + rotation
    barbaserotation = plt.bar(X_axis - 0.2, times_base_rotate, 0.4, label='Rotated time')

    for j in time_overs_base_rotate:
        barbaserotation[j].set_alpha(0.25)

    # SB
    barSB = plt.bar(X_axis + 0.2, times_SB, 0.4, label='Base SB time')

    for j in time_overs_SB:
        barSB[j].set_alpha(0.25)

    # SB + rotation
    barSBrotation = plt.bar(X_axis + 0.6, times_SB_rotate, 0.4, label='Rotated + SB time')

    for j in time_overs_SB_rotate:
        barSBrotation[j].set_alpha(0.25)

    plt.xlabel("VLSI_Instance files")
    plt.ylabel("Time(s)")
    plt.title("VLSI CP Benchmark" + "solver: " + solver_name)
    plt.grid()
    plt.axhline(y=timeout, xmin=0, xmax=1, color='r', linestyle='-.', linewidth=2, label=f"time_limit = {timeout} s")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f'cp_benchmark_{solver_name}.png', transparent=False, format="png")
    plt.show()

    out_text = f"total Base time        -- mean: {np.mean(times_base)} std: {np.std(times_base)}\n"
    out_text += f"total rotated time     -- mean: {np.mean(times_base_rotate)} std: {np.std(times_base_rotate)}\n"
    out_text += f"total SB time          -- mean: {np.mean(times_SB)} std: {np.std(times_SB)}\n"
    out_text += f"total rotated + SB time-- mean: {np.mean(times_SB_rotate)} std: {np.std(times_SB_rotate)}\n"

    print(out_text)
    # write txt log
    with open(f"cp_benchmark_log_{solver_name}.txt", "w") as file:
        content = out_text
        file.writelines(content)
        file.close()


#for i in range(1, 41):
#    cp_benchmark(i, 300, "base", "chuffed", verbose=True, plot=True)

plot_CP_benchmark(40, "chuffed", timeout=300, plot=False)