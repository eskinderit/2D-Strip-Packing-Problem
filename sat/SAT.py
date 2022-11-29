#!pip install z3-solver
import sys
sys.path.append('../')
from utils.utils import *
from z3 import *
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class VLSI_SAT_Instance(VLSI_Instance):
    def load_sat_instance(self, H, rotate=False):
        self.H = H
        self.lr = [[Bool(f"lr_({i},{j})") for j in range(self.n_instances)] for i in range(self.n_instances)]
        self.ud = [[Bool(f"ud_({i},{j})") for j in range(self.n_instances)] for i in range(self.n_instances)]
        self.s = Solver()

        if rotate:
            self.r = [Bool(f"r_({i})") for i in range(self.n_instances)]
            max_len = max(self.W, self.H)
            self.enc_x = [[Bool(f"encx_({i},{j})") for j in range(max_len)] for i in range(self.n_instances)]
            self.enc_y = [[Bool(f"ency_({i},{j})") for j in range(max_len)] for i in range(self.n_instances)]
        else:
            self.enc_x = [[Bool(f"encx_({i},{j})") for j in range(self.W)] for i in range(self.n_instances)]
            self.enc_y = [[Bool(f"ency_({i},{j})") for j in range(self.H)] for i in range(self.n_instances)]


def ordinalencoder(number, max_number):
    if number > max_number:
        raise Exception("The number to encode is bigger than the maximum defined")

    encoded = []

    for i in range(0, number):
        encoded.append(False)

    for i in range(number, max_number):
        encoded.append(True)

    return encoded


def ordinaldecoder(encoded):
    i = 0

    while (i <= (len(encoded) - 1) and not (encoded[i])):
        i = i + 1

    return i


"""
#test
assert 0 == ordinaldecoder(ordinalencoder(0, 4))
assert 1 == ordinaldecoder(ordinalencoder(1, 4))
assert 2 == ordinaldecoder(ordinalencoder(2, 4))
assert 3 == ordinaldecoder(ordinalencoder(3, 4))
assert 4 == ordinaldecoder(ordinalencoder(4, 4))
"""


class VLSI_SAT_solver:
    def domain_constraints(self, instance, break_symmetries=False):

        instances_to_constraint = list(range(instance.n_instances))

        if (break_symmetries):

            biggest_rect_index = instance.biggest_rectangle_index()[0]
            instances_to_constraint.remove(biggest_rect_index)

            for j in range(((instance.H - instance.rectangles[biggest_rect_index].height) // 2), instance.H):
                instance.s.add(instance.enc_y[biggest_rect_index][j])
            for k in range(((instance.W - instance.rectangles[biggest_rect_index].width) // 2), instance.W):
                instance.s.add(instance.enc_x[biggest_rect_index][k])

        for i in instances_to_constraint:
            for j in range(instance.H - instance.rectangles[i].height, instance.H):
                instance.s.add(instance.enc_y[i][j])
            for k in range(instance.W - instance.rectangles[i].width, instance.W):
                instance.s.add(instance.enc_x[i][k])

    def domain_constraints_rotation(self, instance, break_symmetries=False):

        instances_to_constraint = list(range(instance.n_instances))
        r = instance.r

        for i in instances_to_constraint:

            SB_square_i = break_symmetries and (instance.rectangles[i].width == instance.rectangles[i].height)
            non_rotable = (instance.rectangles[i].height > instance.W)

            if (SB_square_i or non_rotable):

                for j in range(instance.H - instance.rectangles[i].height, instance.H):
                    instance.s.add(instance.enc_y[i][j])
                for k in range(instance.W - instance.rectangles[i].width, instance.W):
                    instance.s.add(instance.enc_x[i][k])

            else:

                for j in range(instance.H - instance.rectangles[i].height, instance.H):
                    instance.s.add(Implies(Not(r[i]), instance.enc_y[i][j]))
                for k in range(instance.W - instance.rectangles[i].width, instance.W):
                    instance.s.add(Implies(Not(r[i]), instance.enc_x[i][k]))
                for j in range(instance.H - instance.rectangles[i].width, instance.H):
                    instance.s.add(Implies(r[i], instance.enc_y[i][j]))
                for k in range(instance.W - instance.rectangles[i].height, instance.W):
                    instance.s.add(Implies(r[i], instance.enc_x[i][k]))

    def order_constraints(self, instance):

        for i in range(instance.n_instances):

            for j in range(len(instance.enc_x[i]) - 1):
                instance.s.add(Implies(instance.enc_x[i][j], instance.enc_x[i][j + 1]))

            for k in range(len(instance.enc_y[i]) - 1):
                instance.s.add(Implies(instance.enc_y[i][k], instance.enc_y[i][k + 1]))

    def no_overlap_constraints_rotation(self, instance, break_symmetries=False):
        lr = instance.lr
        ud = instance.ud
        enc_x = instance.enc_x
        enc_y = instance.enc_y
        s = instance.s
        rectangles = instance.rectangles
        H = instance.H
        W = instance.W

        for j in range(instance.n_instances):
            for i in range(j):

                SB_square_i = False
                SB_square_j = False

                if (break_symmetries and (rectangles[i].width == rectangles[i].height)):
                    SB_square_i = True

                if (break_symmetries and (rectangles[j].width == rectangles[j].height)):
                    SB_square_j = True

                relative_positions = []

                relative_positions.append(lr[i][j])

                relative_positions.append(lr[j][i])

                relative_positions.append(ud[i][j])

                relative_positions.append(ud[j][i])

                s.add(Or(relative_positions))

                r = instance.r

                # 1
                if (SB_square_i or (rectangles[i].height > W)):
                    s.add([Or(Not(lr[i][j]), Not(enc_x[j][t])) for t in range(0, rectangles[i].width)])
                    s.add([Or(Not(lr[i][j]), Not(enc_x[j][rectangles[i].width + s]), enc_x[i][s]) for s in
                           range(0, W - rectangles[i].width)])

                else:

                    s.add([Implies(Not(r[i]), (Or(Not(lr[i][j]), Not(enc_x[j][t])))) for t in
                           range(0, rectangles[i].width)])
                    s.add([Implies(Not(r[i]), (Or(Not(lr[i][j]), Not(enc_x[j][rectangles[i].width + s]), enc_x[i][s])))
                           for s in range(0, W - rectangles[i].width)])

                    s.add(
                        [Implies(r[i], (Or(Not(lr[i][j]), Not(enc_x[j][t])))) for t in range(0, rectangles[i].height)])
                    s.add(
                        [Implies(r[i], (Or(Not(lr[i][j]), Not(enc_x[j][rectangles[i].height + s]), enc_x[i][s]))) for s
                         in range(0, W - rectangles[i].height)])

                # 2
                if (SB_square_j or (rectangles[j].height > W)):
                    s.add([Or(Not(lr[j][i]), Not(enc_x[i][t])) for t in range(0, rectangles[j].width)])
                    s.add([Or(Not(lr[j][i]), Not(enc_x[i][rectangles[j].width + s]), enc_x[j][s]) for s in
                           range(0, W - rectangles[j].width)])

                else:

                    s.add([Implies(Not(r[j]), (Or(Not(lr[j][i]), Not(enc_x[i][t])))) for t in
                           range(0, rectangles[j].width)])
                    s.add([Implies(Not(r[j]), (Or(Not(lr[j][i]), Not(enc_x[i][rectangles[j].width + s]), enc_x[j][s])))
                           for s in range(0, W - rectangles[j].width)])

                    s.add(
                        [Implies(r[j], (Or(Not(lr[j][i]), Not(enc_x[i][t])))) for t in range(0, rectangles[j].height)])
                    s.add(
                        [Implies(r[j], (Or(Not(lr[j][i]), Not(enc_x[i][rectangles[j].height + s]), enc_x[j][s]))) for s
                         in range(0, W - rectangles[j].height)])

                # 3

                if (SB_square_i or (rectangles[i].height > W)):
                    s.add([Or(Not(ud[i][j]), Not(enc_y[j][t])) for t in range(0, rectangles[i].height)])
                    s.add([Or(Not(ud[i][j]), Not(enc_y[j][rectangles[i].height + s]), enc_y[i][s]) for s in
                           range(0, H - rectangles[i].height)])

                else:

                    s.add([Implies(Not(r[i]), (Or(Not(ud[i][j]), Not(enc_y[j][t])))) for t in
                           range(0, rectangles[i].height)])
                    s.add([Implies(Not(r[i]), (Or(Not(ud[i][j]), Not(enc_y[j][rectangles[i].height + s]), enc_y[i][s])))
                           for s in range(0, H - rectangles[i].height)])

                    s.add([Implies(r[i], (Or(Not(ud[i][j]), Not(enc_y[j][t])))) for t in range(0, rectangles[i].width)])
                    s.add([Implies(r[i], (Or(Not(ud[i][j]), Not(enc_y[j][rectangles[i].width + s]), enc_y[i][s]))) for s
                           in range(0, H - rectangles[i].width)])

                # 4

                if (SB_square_j or (rectangles[j].height > W)):
                    s.add([Or(Not(ud[j][i]), Not(enc_y[i][t])) for t in range(0, rectangles[j].height)])
                    s.add([Or(Not(ud[j][i]), Not(enc_y[i][rectangles[j].height + s]), enc_y[j][s]) for s in
                           range(0, H - rectangles[j].height)])

                else:

                    s.add([Implies(Not(r[j]), (Or(Not(ud[j][i]), Not(enc_y[i][t])))) for t in
                           range(0, rectangles[j].height)])
                    s.add([Implies(Not(r[j]), (Or(Not(ud[j][i]), Not(enc_y[i][rectangles[j].height + s]), enc_y[j][s])))
                           for s in range(0, H - rectangles[j].height)])

                    s.add([Implies(r[j], (Or(Not(ud[j][i]), Not(enc_y[i][t])))) for t in range(0, rectangles[j].width)])
                    s.add([Implies(r[j], (Or(Not(ud[j][i]), Not(enc_y[i][rectangles[j].width + s]), enc_y[j][s]))) for s
                           in range(0, H - rectangles[j].width)])

    def no_overlap_constraints(self, instance, break_symmetries=False):

        lr = instance.lr
        ud = instance.ud
        enc_x = instance.enc_x
        enc_y = instance.enc_y
        s = instance.s
        rectangles = instance.rectangles
        H = instance.H
        W = instance.W
        biggest_rect_index = instance.biggest_rectangle_index()

        for j in range(instance.n_instances):
            for i in range(j):

                # symmetry breaking constraints
                SB_large_rectangles_x = False
                SB_large_rectangles_y = False
                SB_same_rectangles = False
                SB_biggest_rect = False

                if (break_symmetries and ((rectangles[i].width + rectangles[j].width) > W)):
                    SB_large_rectangles_x = True

                if (break_symmetries and ((rectangles[i].height + rectangles[j].height) > H)):
                    SB_large_rectangles_y = True

                if (break_symmetries and (rectangles[i].height == rectangles[j].height) and (
                        rectangles[i].width == rectangles[j].width)):
                    SB_same_rectangles = True
                    s.add(Implies(ud[i][j], lr[j][i]))

                if (break_symmetries and j == biggest_rect_index):
                    SB_biggest_rect = True

                relative_positions = []

                if ((not SB_large_rectangles_x) and (
                not (SB_biggest_rect and (rectangles[i].width > (W - rectangles[biggest_rect_index].width) // 2)))):
                    relative_positions.append(lr[i][j])

                if ((not SB_large_rectangles_x) and (not SB_same_rectangles)):
                    relative_positions.append(lr[j][i])

                if ((not SB_large_rectangles_y) and (
                not (SB_biggest_rect and (rectangles[i].height > (H - rectangles[biggest_rect_index].height) // 2)))):
                    relative_positions.append(ud[i][j])

                if ((not SB_large_rectangles_y)):
                    relative_positions.append(ud[j][i])

                s.add(Or(relative_positions))

                # 1
                if ((not SB_large_rectangles_x) and (
                not (SB_biggest_rect and (rectangles[i].width > (W - rectangles[biggest_rect_index].width) // 2)))):
                    s.add([Or(Not(lr[i][j]), Not(enc_x[j][t])) for t in range(0, rectangles[i].width)])
                    s.add([Or(Not(lr[i][j]), Not(enc_x[j][rectangles[i].width + s]), enc_x[i][s]) for s in
                           range(0, W - rectangles[i].width)])

                # 2
                if ((not SB_large_rectangles_x) and (not SB_same_rectangles)):
                    s.add([Or(Not(lr[j][i]), Not(enc_x[i][t])) for t in range(0, rectangles[j].width)])
                    s.add([Or(Not(lr[j][i]), Not(enc_x[i][rectangles[j].width + s]), enc_x[j][s]) for s in
                           range(0, W - rectangles[j].width)])

                # 3
                if ((not SB_large_rectangles_y) and (
                not (SB_biggest_rect and (rectangles[i].height > (H - rectangles[biggest_rect_index].height) // 2)))):
                    s.add([Or(Not(ud[i][j]), Not(enc_y[j][t])) for t in range(0, rectangles[i].height)])
                    s.add([Or(Not(ud[i][j]), Not(enc_y[j][rectangles[i].height + s]), enc_y[i][s]) for s in
                           range(0, H - rectangles[i].height)])

                # 4
                if ((not SB_large_rectangles_y)):
                    s.add([Or(Not(ud[j][i]), Not(enc_y[i][t])) for t in range(0, rectangles[j].height)])
                    s.add([Or(Not(ud[j][i]), Not(enc_y[i][rectangles[j].height + s]), enc_y[j][s]) for s in
                           range(0, H - rectangles[j].height)])

    def build_constraints_solve(self, instance, break_symmetries=False, rotate=False):
        '''
        Applying constraints depending on the chosen mechanism.
        Returns the solver elapsed time and the satisfiability outcome
        '''

        self.order_constraints(instance)

        if rotate:
            self.domain_constraints_rotation(instance, break_symmetries)
            self.no_overlap_constraints_rotation(instance, break_symmetries=break_symmetries)
        else:
            self.domain_constraints(instance, break_symmetries)
            self.no_overlap_constraints(instance, break_symmetries=break_symmetries)

        satisfiable = (instance.s.check() == sat)
        execution_time = instance.s.statistics().time

        return satisfiable, execution_time

    def solve(self, instance_path, timeout=300, break_symmetries=False, rotate=False, verbose=True, plot=False,
              log_txt=True):
        '''
        Solving strategy: bisection method

        instance_path: the instance folder path
        break_symmetries: True if the breaking symmetry constraints have to be applied
        rotate: True if the rectangles can be rotated
        verbose: True to print the solution in a verbose output mode
        plot: True to print the graphic solution
        '''

        time_over = False
        z3_total_time = 0

        set_option(timeout=timeout * 1000)

        instance = VLSI_SAT_Instance(instance_path)
        rectangles = instance.rectangles

        # setting Lower Bound and Upper Bound

        LB_init = instance.H_LB()

        if rotate:
            UB_init = instance.H_UB_rotation()
        else:
            UB_init = instance.H_UB_BL()

        if verbose:
            print("### Height LB: {}, Height UB: {}, N_rectangles: {} ###".format(LB_init, UB_init, instance.n_instances))

        # using bisection method as search strategy
        start = time.time()
        best_bound = False
        solution_found = "UPPER_BOUND"
        LB = LB_init
        UB = UB_init

        # for "fairness", we launch the solver even if the LB has already reached the best bound
        while LB <= UB and not best_bound:

            o = (LB + UB) // 2
            instance.load_sat_instance(H=o, rotate=rotate)

            if verbose:
                print("Attempting Height = ", o)

            solved, timer = self.build_constraints_solve(instance, break_symmetries=break_symmetries, rotate=rotate)
            z3_total_time += timer


            if verbose:
                print("Attempting Height = ", o, ", Elapsed time: ", timer)

            if solved:
                print("solved branch")
                solution_found = "NOT_OPTIMAL"

                model = copy.copy(instance.s.model())
                enc_x = copy.copy(instance.enc_x)
                enc_y = copy.copy(instance.enc_y)
                UB = o

                if (verbose):
                    print("success with Height = ", o)

                if LB == UB:
                    best_bound = True


            else:
                print("else branch")
                # if the solving process failed, the solve is attempted with
                # the other bisection extreme

                if verbose:
                    print("fail with Height = ", o)

                LB = o + 1

        end = time.time()
        rectangle_placements = []

        if z3_total_time + 0.1 >= timeout:
            time_over = True
            z3_total_time = 301


        # here we assign the new values of the solution just if the solver has found something
        # different from the initially computed upper bound
        if not(solution_found == "UPPER_BOUND"):

            for i in range(len(rectangles)):

                # inverting side measures for rotated rectangles
                if rotate:
                    if model.__getitem__(instance.r[i]):
                        w = rectangles[i].width
                        h = rectangles[i].height
                        rectangles[i].width = h
                        rectangles[i].height = w

                # assigning the computed left-bottom corner coordinates to rectangles
                x = []
                y = []

                for j in range(len(enc_x[i])):
                    x.append(model.__getitem__(enc_x[i][j]))

                for j in range(len(enc_y[i])):
                    y.append(model.__getitem__(enc_y[i][j]))

                rectangles[i].x = ordinaldecoder(x)
                rectangles[i].y = ordinaldecoder(y)

                rotated = "No"

                if (rotate and model.__getitem__(instance.r[i])):
                    rotated = "Yes"

                rectangle_placement = "coordinate for rectangle[{}]: (x: ,{}, y: {}), width: {}, height: {}, rotated: {} \n".format(
                    i, rectangles[i].x, rectangles[i].y, rectangles[i].width, rectangles[i].height, rotated)

                # printing computed parameters

                if (verbose):
                    print(rectangle_placement)

                rectangle_placements.append(rectangle_placement)

            # computing the maximum height that succeeded
            instance.H = max([(r.y + r.height) for r in rectangles])

        total_time = end - start

        if instance.H == LB_init:
            solution_found = "OPTIMAL"
        if plot:
            plot_rectangles(rectangles, title=instance.name)

        if verbose:
            print(f"### Best height: {instance.H}, Computation time: {total_time}s ###")

        if log_txt:
            if rotate:
                if break_symmetries:
                    path = "rotations-sb"
                else:
                    path = "rotations"
            else:
                if break_symmetries:
                    path = "base-sb"
                else:
                    path = "base"

            title_log_txt = title = re.split("/", instance.name)[-1]

            write_log(path="../out/sat/" + path + "/" + title_log_txt, instance=instance,
                      add_text="\n" + str(z3_total_time) + "\n" + str(time_over)+ "\n" + solution_found)

        return rectangles, instance.H, total_time, time_over, z3_total_time


def plot_SAT_benchmark(instances_to_solve=5, timeout=300, plot=False, verbose=False):
    '''
    Plotting the barplot with all the SAT solving mechanisms (base, rotated,
    base + symmetry breaking, rotated + symmetry breaking)
    '''

    times_base = []
    times_base_rotate = []
    times_SB = []
    times_SB_rotate = []

    z3_times_base = []
    z3_times_base_rotate = []
    z3_times_SB = []
    z3_times_SB_rotate = []

    time_overs_base = []
    time_overs_base_rotate = []
    time_overs_SB = []
    time_overs_SB_rotate = []

    for i in tqdm(range(1, instances_to_solve + 1)):

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = VLSI_SAT_solver().solve(instance_path=url, timeout=timeout,
                                                                   break_symmetries=False, rotate=False, verbose=verbose,
                                                                   plot=plot)
        times_base.append(timer)
        z3_times_base.append(z3_timer)

        if (time_over):
            time_overs_base.append(i - 1)

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = VLSI_SAT_solver().solve(instance_path=url, timeout=timeout,
                                                                   break_symmetries=False, rotate=True, verbose=verbose,
                                                                   plot=plot)
        times_base_rotate.append(timer)
        z3_times_base_rotate.append(z3_timer)

        if (time_over):
            time_overs_base_rotate.append(i - 1)

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = VLSI_SAT_solver().solve(instance_path=url, timeout=timeout,
                                                                   break_symmetries=True, rotate=False, verbose=verbose,
                                                                   plot=plot)
        times_SB.append(timer)
        z3_times_SB.append(z3_timer)

        if (time_over):
            time_overs_SB.append(i - 1)

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = VLSI_SAT_solver().solve(instance_path=url, timeout=timeout,
                                                                   break_symmetries=True, rotate=True, verbose=verbose,
                                                                   plot=plot)
        times_SB_rotate.append(timer)
        z3_times_SB_rotate.append(z3_timer)

        if (time_over):
            time_overs_SB_rotate.append(i - 1)

    X = range(1, instances_to_solve + 1)
    X_axis = np.arange(0, len(times_SB) * 2, 2)

    plt.rcParams["figure.figsize"] = (13, 6)
    plt.xticks(X_axis, X)

    # base z3
    barbase = plt.bar(X_axis - 0.6, z3_times_base, 0.4, label='Base z3 time')

    # base total
    plt.plot(X_axis, times_base, linestyle='--', marker='o', label='Base total time')

    for i in time_overs_base:
        barbase[i].set_alpha(0.25)

    # base + rotation z3
    barbaserotation = plt.bar(X_axis - 0.2, z3_times_base_rotate, 0.4, label='Rotated z3 time')

    # base + rotation total
    plt.plot(X_axis, times_base_rotate, linestyle='--', marker='o', label='Rotated total time')

    for i in time_overs_base_rotate:
        barbaserotation[i].set_alpha(0.25)

    # SB
    barSB = plt.bar(X_axis + 0.2, z3_times_SB, 0.4, label='SB z3 time')

    # SB total
    plt.plot(X_axis, times_SB, linestyle='--', marker='o', label='SB total time')

    for i in time_overs_SB:
        barSB[i].set_alpha(0.25)

    # SB + rotation
    barSBrotation = plt.bar(X_axis + 0.6, z3_times_SB_rotate, 0.4, label='Rotated + SB z3 time')

    # SB + rotation total
    plt.plot(X_axis, times_SB_rotate, linestyle='--', marker='o', label='Rotated + SB total time')

    for i in time_overs_SB_rotate:
        barSBrotation[i].set_alpha(0.25)

    plt.xlabel("VLSI instance files")
    plt.ylabel("Time(s)")
    plt.title("VLSI SAT Benchmark")
    plt.grid()
    plt.axhline(y=timeout, xmin=0, xmax=1, color='r', linestyle='-.', linewidth=2, label=f"time_limit = {timeout} s")
    # plt.yscale("log")
    plt.ylim([0, 350])
    plt.legend()
    plt.savefig('sat_benchmark.png', transparent=False, format="png")
    plt.show()

    out_text = ""

    out_text += f"total Base time        -- mean: {np.mean(times_base)} std: {np.std(times_base)} \n"
    out_text += f"total rotated time     -- mean: {np.mean(times_base_rotate)} std: {np.std(times_base_rotate)} \n"
    out_text += f"total SB time          -- mean: {np.mean(times_SB)} std: {np.std(times_SB)} \n"
    out_text += f"total rotated + SB time-- mean: {np.mean(times_SB_rotate)} std: {np.std(times_SB_rotate)} \n"

    out_text += f"z3 Base time           -- mean: {np.mean(z3_times_base)} std: {np.std(z3_times_base)} \n"
    out_text += f"z3 rotated time        -- mean: {np.mean(z3_times_base_rotate)} std: {np.std(z3_times_base_rotate)} \n"
    out_text += f"z3 SB time             -- mean: {np.mean(z3_times_SB)} std: {np.std(z3_times_SB)} \n"
    out_text += f"z3 rotated + SB time   -- mean: {np.mean(z3_times_SB_rotate)} std: {np.std(z3_times_SB_rotate)}"

    # write txt log
    with open("sat_benchmark_log.txt", "w") as file:
        content = out_text
        file.writelines(content)
        file.close()

    print(out_text)


# timeout is set in seconds
plot_SAT_benchmark(instances_to_solve=40, timeout=300, plot=True)