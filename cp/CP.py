import sys
import minizinc
from datetime import timedelta
from tqdm import tqdm
from minizinc import Instance, Model, Solver

sys.path.append('../')
from utils.utils import *


def cp_benchmark(index, timeout, method, solver_name, verbose=True, plot=True):
    """
    performs a benchmark in which instances having index from
    start to end are tested
    """

    timeout_ = timedelta(seconds=timeout)

    folder = "../instances/"
    file = f"ins-{index}.txt"
    url = folder + file
    cp_instance = VLSI_Instance(url)
    width, height = cp_instance.get_width_height()

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

    mzn_instance["h_ub"] = cp_instance.H_UB_BL()
    mzn_instance["fixed_width"] = cp_instance.W
    mzn_instance["n_components"] = cp_instance.n_instances
    mzn_instance["heights"] = height
    mzn_instance["widths"] = width

    time_over = None
    solution_found = None

    try:
        # solve
        result = mzn_instance.solve(timeout=timeout_, random_seed=7, all_solutions=False,intermediate_solutions=False,free_search=True)

        if verbose:
            print(result)
        solve_time = result.statistics["solveTime"].total_seconds()


        if result.solution is None:
            time_over = True
            solution_found = "UPPER_BOUND"
            solve_time = timeout + 1
            if verbose:
                print(f"instance {index} not solved")

        else:

            if round(solve_time + 0.6, 0) >= timeout: #precision on time bound depends on solver, for chuffed~0.6
                time_over = True
                solution_found = "NOT_OPTIMAL"
                solve_time = timeout + 1
                if verbose:
                    print(f"instance {index} solved, but solution is not optimal")
            else:
                time_over = False
                solution_found = "OPTIMAL"
                if verbose:
                    print(f"Instance {index} solved in ", solve_time)

            cp_instance.H = result.solution.height
            x_pos = result.solution.x_coords
            y_pos = result.solution.y_coords

            if method == "rotations" or method == "rotations-sb":
                rotated = result.solution.rot
                for i in range(0, cp_instance.n_instances):
                    if rotated[i] == 1:
                        new_width = cp_instance.rectangles[i].height
                        new_height = cp_instance.rectangles[i].width
                        cp_instance.rectangles[i].width = new_width
                        cp_instance.rectangles[i].height = new_height

            for i in range(0, cp_instance.n_instances):
                cp_instance.rectangles[i].x = x_pos[i]
                cp_instance.rectangles[i].y = y_pos[i]

            if plot:
                plot_rectangles(cp_instance.rectangles, url)

    except minizinc.error.MiniZincError:
        time_over = True
        solution_found = "UPPER_BOUND"
        solve_time = timeout + 1
        if verbose:
            print(f"instance {index} not solved")

    path = method + "/" + solver_name
    write_log(path="../out/cp/" + path + "/" + file, instance=cp_instance,
              add_text="\n" + str(solve_time) + "\n" + str(time_over) + "\n" + solution_found)
    return solve_time, time_over


def plot_CP_benchmark(instances_to_solve: int = 40, solver_name: str = "chuffed", timeout: int =300, plot=False):
    """
    Produces the barplot with all the CP solving mechanisms (base, rotations,
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
    #plt.yscale("log")
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


#plot_result("../out/cp/base/chuffed/ins-1.txt")
read_reached_bounds("../out/cp/base/chuffed",1,40)
# uncomment one of the two lines following this one to do a complete benchmark on all the 40 sample instances
#plot_CP_benchmark(instances_to_solve=40,  solver_name="chuffed", timeout=300, plot=True)
#plot_CP_benchmark(instances_to_solve=40,  solver_name="or-tools", timeout=300, plot=True)
