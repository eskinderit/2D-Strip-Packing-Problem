from gurobipy import *
import sys

sys.path.append('../')
from utils.utils import *
from datetime import timedelta
from tqdm import tqdm


def lp_benchmark(index, timeout, method, solver_name, verbose=True, plot=True):
    """
    performs a benchmark in which instances having index from
    start to end are tested
    """

    folder = "../instances/"
    file = f"ins-{index}.txt"
    url = folder + file
    lp_instance = VLSI_Instance(path=url, order_by_width=True)
    width, height = lp_instance.get_width_height()

    # Load 2d-strip-packaging model from file
    '''
    if method == "rotations":
        model = Model("./LP_rotated.mzn")
    elif method == "rotations-sb":
        model = Model("./LP_rotated_SB.mzn")
    elif method == "base-sb":
        model = Model("./LP_sb.mzn")
    elif method == "base":
        model = Model("./LP.mzn")
    else:
        model = Model("./LP.mzn")
    '''

    # Create an Instance of the 2d-strip-packaging model
    gurobi_model = Model()

    if method =="rotations":
        gurobi_model = Model("lp-rotations")
        gurobi_model.addVars("biggest_dim", vtype=GRB.INTEGER)
    if method == "rotations-sb":
        gurobi_model = Model("lp-rotations-sb")
        squares = lp_instance.get_squares_index()
        gurobi_model["n_squares"] = len(squares)
        gurobi_model["square_index"] = squares
        gurobi_model["biggest_dim"] = max(max(height), max(width))


    if method == "base-sb":
        large_rectangles = lp_instance.get_large_rectangles_index()
        same_dim_rectangles = lp_instance.get_same_dim_rectangles_index()
        biggest_rectangle, smaller_rectangles = lp_instance.biggest_rectangle_index()
        gurobi_model["n_large_rectangles"] = len(large_rectangles)
        gurobi_model["large_rectangles"] = large_rectangles
        gurobi_model["n_same_dim_rectangles"] = len(same_dim_rectangles)
        gurobi_model["same_dim_rectangles"] = same_dim_rectangles
        gurobi_model["biggest_rect"] = biggest_rectangle
        gurobi_model["n_smaller_rectangles"] = len(smaller_rectangles)
        gurobi_model["smaller_rectangles"] = smaller_rectangles

    n_instances = lp_instance.n_instances
    W = lp_instance.W
    H_UB = lp_instance.H_UB_BL()
    H_LB = lp_instance.H_LB()
    H = gurobi_model.addVar(lb= H_LB, ub=H_UB, vtype=GRB.INTEGER, name="H")
    #delta = [[[0]*2]*n_instances]*n_instances

    delta = gurobi_model.addVars(n_instances, n_instances, [i for i in range(0, 2)], lb=0, ub=1, vtype=GRB.BINARY, name="delta")
    print(delta)

    x = [0]*n_instances
    y = [0]*n_instances

    for i in range(n_instances):
        x[i] = gurobi_model.addVar(lb=0, ub= W-width[i], vtype=GRB.INTEGER, name="x")
        y[i] = gurobi_model.addVar( lb=0, ub= H_UB - height[i], vtype=GRB.INTEGER, name="y")
        gurobi_model.addLConstr(y[i] <= (H - height[i]))

        for j in range(n_instances):

            if j < i:
                print(type(delta))
                gurobi_model.addLConstr(x[i] + width[i] <= x[j] + (W * delta[i, j, 0]))
                gurobi_model.addLConstr(x[j] + width[j] <= x[i] + (W * delta[j, i, 0]))
                gurobi_model.addLConstr(y[i] + height[i] <= y[j] + (H_UB * delta[i, j, 1]))
                gurobi_model.addLConstr(y[j] + height[j] <= y[i] + (H_UB * delta[j, i, 1]))
                gurobi_model.addLConstr(delta[i, j, 0] + delta[i, j, 1] + delta[j, i, 0] + delta[j, i, 1] <= 3)



    # solve
    gurobi_model.setObjective(H, GRB.MINIMIZE)
    gurobi_model.setParam('MIPFocus', 2)
    #gurobi_model.write('model.lp')
    #gurobi_model.setParam('Symmetry',2)
    #gurobi_model.setParam('ConcurrentMIP', 2)
    #gurobi_model.setParam('SolutionNumber', 1)
    #gurobi_model.setParam('TimeLimit', timeout)
    gurobi_model.setParam('Seed', 8)
    gurobi_model.optimize()

    if verbose:
        print(gurobi_model.getVars())

    solve_time = gurobi_model.Runtime

    time_over = False

    if solve_time >= timeout or (gurobi_model.getVars() is None):
        time_over = True
        solve_time = 301

        if verbose:
            print(f"instance {index} overtime")
    else:
        if verbose:
            print(f"Instance {index} solved in ", solve_time)

        if plot:
            # plot results

            xy_pos = [int(i) for i in gurobi_model.getAttr('X', (x[2],y[2]))]
            H = int(gurobi_model.getAttr('X', (H,))[0])
            print(xy_pos)
            #print(y_pos)
            """
            if method == "rotations" or method == "rotations-sb":
                rotated = result.solution.rotated
                for i in range(0, lp_instance.n_instances):
                    if rotated[i] == 1:
                        new_width = lp_instance.rectangles[i].height
                        new_height = lp_instance.rectangles[i].width
                        lp_instance.rectangles[i].width = new_width
                        lp_instance.rectangles[i].height = new_height
            """
            for i in range(0, lp_instance.n_instances):
                lp_instance.rectangles[i].x = x[i].X
                lp_instance.rectangles[i].y = y[i].X

            lp_instance.H = H
            plot_rectangles(lp_instance.rectangles, url)
    """
    path = method + "/" + solver_name
    write_log(path="../out/lp/" + path + "/" + file, instance=lp_instance,
              add_text="\n" + str(solve_time) + "\n" + str(time_over))
    """
    return solve_time, time_over


def plot_LP_benchmark(instances_to_solve: int = 40, solver_name: str = "gurobi", timeout: int = 300, plot=False):
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
        time, time_over = lp_benchmark(j, timeout=timeout, method="base", solver_name=solver_name, plot=plot)
        times_base.append(time)

        if time_over:
            time_overs_base.append(j - 1)

        # base + SB
        time, time_over = lp_benchmark(j, timeout=timeout, method="base-sb", solver_name=solver_name, plot=plot)
        times_SB.append(time)

        if time_over:
            time_overs_SB.append(j - 1)

        # rotated
        time, time_over = lp_benchmark(j, timeout=timeout, method="rotations", solver_name=solver_name, plot=plot)
        times_base_rotate.append(time)

        if time_over:
            time_overs_base_rotate.append(j - 1)

        # rotated + SB
        time, time_over = lp_benchmark(j, timeout=timeout, method="rotations-sb", solver_name=solver_name, plot=plot)
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
    plt.title("VLSI LP Benchmark" + "solver: " + solver_name)
    plt.grid()
    plt.axhline(y=timeout, xmin=0, xmax=1, color='r', linestyle='-.', linewidth=2, label=f"time_limit = {timeout} s")
    #plt.yscale("log")
    plt.legend()
    plt.savefig(f'lp_benchmark_{solver_name}.png', transparent=False, format="png")
    plt.show()

    out_text = f"total Base time        -- mean: {np.mean(times_base)} std: {np.std(times_base)}\n"
    out_text += f"total rotated time     -- mean: {np.mean(times_base_rotate)} std: {np.std(times_base_rotate)}\n"
    out_text += f"total SB time          -- mean: {np.mean(times_SB)} std: {np.std(times_SB)}\n"
    out_text += f"total rotated + SB time-- mean: {np.mean(times_SB_rotate)} std: {np.std(times_SB_rotate)}\n"

    print(out_text)
    # write txt log
    with open(f"lp_benchmark_log_{solver_name}.txt", "w") as file:
        content = out_text
        file.writelines(content)
        file.close()


#for i in range(31, 40):
lp_benchmark(26, 300, "base", "gurobi", plot=True)
# timeout is set in seconds
#plot_LP_benchmark(instances_to_solve=40, solver_name="gurobi", timeout=300, plot=True)

