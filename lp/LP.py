import sys

from minizinc import Instance, Model, Solver

sys.path.append('../')
from utils.utils import *
from datetime import timedelta
from tqdm import tqdm


def lp_benchmark(index, timeout, method, solver_name, verbose=True, plot=True):
    """
    performs a benchmark in which instances having index from
    start to end are tested
    """

    timeout_ = timedelta(seconds=timeout)

    folder = "../instances/"
    file = f"ins-{index}.txt"
    url = folder + file
    lp_instance = VLSI_Instance(path=url, order_by_width=True)
    width, height = lp_instance.get_width_height()

    # Load 2d-strip-packaging model from file
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

    # Find the MiniZinc solver configuration
    solver = Solver.lookup(solver_name)

    # Create an Instance of the 2d-strip-packaging model
    mzn_instance = Instance(solver, model)

    if method =="rotations":
        mzn_instance["biggest_dim"] = max(max(height), max(width))
    if method == "rotations-sb":
        squares = lp_instance.get_squares_index()
        mzn_instance["n_squares"] = len(squares)
        mzn_instance["square_index"] = squares
        mzn_instance["biggest_dim"] = max(max(height), max(width))


    if method == "base-sb":
        large_rectangles = lp_instance.get_large_rectangles_index()
        same_dim_rectangles = lp_instance.get_same_dim_rectangles_index()
        biggest_rectangle, smaller_rectangles = lp_instance.biggest_rectangle_index()
        mzn_instance["n_large_rectangles"] = len(large_rectangles)
        mzn_instance["large_rectangles"] = large_rectangles
        mzn_instance["n_same_dim_rectangles"] = len(same_dim_rectangles)
        mzn_instance["same_dim_rectangles"] = same_dim_rectangles
        mzn_instance["biggest_rect"] = biggest_rectangle
        mzn_instance["n_smaller_rectangles"] = len(smaller_rectangles)
        mzn_instance["smaller_rectangles"] = smaller_rectangles

    mzn_instance["h_ub"] = lp_instance.H_UB_BL()
    mzn_instance["h_lb"] = lp_instance.H_LB()
    mzn_instance["W"] = lp_instance.W
    mzn_instance["n_rectangles"] = lp_instance.n_instances
    mzn_instance["rect_height"] = height
    mzn_instance["rect_width"] = width

    # solve
    result = mzn_instance.solve(timeout=timeout_, random_seed=8)
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
            x_pos = result.solution.rect_x
            y_pos = result.solution.rect_y
            H = result.solution.H

            if method == "rotations" or method == "rotations-sb":
                rotated = result.solution.rotated
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

    path = method + "/" + solver_name
    write_log(path="../out/lp/" + path + "/" + file, instance=lp_instance,
              add_text="\n" + str(solve_time) + "\n" + str(time_over))
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
#    lp_benchmark(i, 300, "base", "gurobi")
# timeout is set in seconds
plot_LP_benchmark(instances_to_solve=40, solver_name="gurobi", timeout=300, plot=True)
