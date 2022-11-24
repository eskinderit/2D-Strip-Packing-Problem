import sys
from ortools.linear_solver import pywraplp

sys.path.append('../')
from utils.utils import *
from tqdm import tqdm


def build_common_constraints(solver, lp_instance):
    n_instances = lp_instance.n_instances
    H_UB = lp_instance.H_UB_BL()
    H_LB = lp_instance.H_LB()
    H = solver.IntVar(H_LB, H_UB, name="H")
    delta = [[[0] * 4] * n_instances] * n_instances

    def generate_vars(x, y, z):
        if x != y:
            return solver.BoolVar(name=f"delta_{x, y, z}")
        else:
            return None

    for i in range(n_instances):
        delta[i] = [[generate_vars(i, j, k) for k in range(2)] for j in range(n_instances)]

    return H, delta

def build_rotate_constraints(solver, lp_instance, delta, H):
    W = lp_instance.W
    H_UB = lp_instance.H_UB_BL()
    width, height = lp_instance.get_width_height()
    n_instances = lp_instance.n_instances
    r = [0] * n_instances
    new_width = [0] * n_instances
    new_height = [0] * n_instances
    x = [0] * n_instances
    y = [0] * n_instances

    for i in range(n_instances):
        r[i] = solver.IntVar(0, 1, name=f"r{i}")
        new_width[i] = solver.IntVar(0, W, name=f"new_width{i}")
        new_height[i] = solver.IntVar(0, H_UB, name=f"new_height{i}")

    for i in range(n_instances):
        x[i] = solver.IntVar(0, W, name=f"x{i}")
        y[i] = solver.IntVar(0, H_UB, name=f"y{i}")
        solver.Add(x[i] <= W - new_width[i])
        solver.Add(y[i] <= H - new_height[i])

    for i in range(n_instances):
        for j in range(n_instances):
            if j < i:
                solver.Add(x[i] + new_width[i] <= x[j] + W * delta[i][j][0])
                solver.Add(x[j] + new_width[j] <= x[i] + W * delta[j][i][0])
                solver.Add(y[i] + new_height[i] <= y[j] + H_UB * delta[i][j][1])
                solver.Add(y[j] + new_height[j] <= y[i] + H_UB * delta[j][i][1])
                solver.Add(sum([delta[i][j][0], delta[i][j][1], delta[j][i][0], delta[j][i][1]]) <= 3)

    for i in range(n_instances):
        solver.Add(new_width[i] == r[i] * height[i] + (1 - r[i]) * width[i])
        solver.Add(new_height[i] == r[i] * width[i] + (1 - r[i]) * height[i])

    return x, y, r

def build_base_constraints(solver, lp_instance, delta, H):
    width, height = lp_instance.get_width_height()
    n_instances = lp_instance.n_instances
    W = lp_instance.W
    H_UB = lp_instance.H

    x = [0] * n_instances
    y = [0] * n_instances

    for i in range(n_instances):
        x[i] = solver.IntVar(0, W, name=f"x{i}")
        y[i] = solver.IntVar(0, H_UB, name=f"y{i}")
        solver.Add(x[i] <= W - width[i])
        solver.Add(y[i] <= H - height[i])


    for i in range(n_instances):
        for j in range(n_instances):
            if j < i:
                solver.Add(x[i] + width[i] <= x[j] + W * delta[i][j][0])
                solver.Add(x[j] + width[j] <= x[i] + W * delta[j][i][0])
                solver.Add(y[i] + height[i] <= y[j] + H_UB * delta[i][j][1])
                solver.Add(y[j] + height[j] <= y[i] + H_UB * delta[j][i][1])
                solver.Add(sum([delta[i][j][0], delta[i][j][1], delta[j][i][0], delta[j][i][1]]) <= 3)

    return x, y

def build_sb_constraints(solver, lp_instance, delta, x, y, H):

    # constraint for the symmetry breaking of large rectangles
    # (those such that width[i] + width[j] > W)

    large_rectangles = [np.subtract(i, [1, 1], dtype=np.int64) for i in lp_instance.get_large_rectangles_index()]
    for i in large_rectangles:
        solver.Add(delta[i[0]][i[1]][0] == 1)
        solver.Add(delta[i[1]][i[0]][0] == 1)

    # constraint for the symmetry breaking of large rectangles
    # (those such that height[i] = height[j] and width[i] = width[j])
    same_dim_rectangles = [np.subtract(i, [1, 1], dtype=np.int64) for i in lp_instance.get_same_dim_rectangles_index()]
    for i in same_dim_rectangles:
        # rectangle i can't be above rectangle j
        solver.Add(delta[i[1]][i[0]][0] == 1)
        # rectangle i can't be right to retangle j
        solver.Add(delta[i[1]][i[0]][1] == 1)

    W = lp_instance.W
    biggest_rectangle, smaller_rectangles = lp_instance.biggest_rectangle_index()
    biggest_rectangle = biggest_rectangle - 1
    smaller_rectangles = [i-1 for i in smaller_rectangles]
    solver.Add(2 * x[biggest_rectangle] <= W - lp_instance.rectangles[biggest_rectangle].width)
    solver.Add(2 * y[biggest_rectangle] <= H - lp_instance.rectangles[biggest_rectangle].height)
    for i in smaller_rectangles:
        # with this constraint we make sure that
        # smaller rectangles (satisfying conditions defined in the utils)
        # will go right, over or under the biggest but never left
        solver.Add(delta[biggest_rectangle][i][0] + delta[i][biggest_rectangle][1] + delta[biggest_rectangle][i][1] <= 2)
        solver.Add(delta[biggest_rectangle][i][0] + delta[i][biggest_rectangle][1] + delta[biggest_rectangle][i][1] >= 1)


def build_rotations_sb_constraints(solver, lp_instance, r):
    squares = [i-1 for i in lp_instance.get_squares_index()]
    for i in squares:
        solver.Add(r[i] == 0)

def lp_benchmark(index, timeout, method, solver_name, verbose=True, plot=True):
    """
    performs a benchmark in which instances having index from
    start to end are tested
    """
    if solver_name == "gurobi":
        solver_tag = pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING
    if solver_name == "scip":
        solver_tag = pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING

    folder = "../instances/"
    file = f"ins-{index}.txt"
    url = folder + file
    lp_instance = VLSI_Instance(path=url, order_by_width=True)

    # solver settings
    #solver = pywraplp.Solver.CreateSolver(solver_name)
    solver = pywraplp.Solver('SolveIntegerProblem',solver_tag)
    solver_parameters = pywraplp.MPSolverParameters()
    solver_parameters.SetDoubleParam(pywraplp.MPSolverParameters.PRIMAL_TOLERANCE, 0.001)
    #solver_parameters.SetDoubleParam(pywraplp.MPSolverParameters.DUAL_TOLERANCE, 0.001)
    #print(solver_parameters.GetDoubleParam(pywraplp.MPSolverParameters.PRIMAL_TOLERANCE))
    if not solver:
        print("no solver found")

    H, delta = build_common_constraints(solver, lp_instance)
    width, height = lp_instance.get_width_height()

    if method == "base":
        x, y = build_base_constraints(solver, lp_instance, delta, H)
    if method == "base-sb":
        x, y = build_base_constraints(solver, lp_instance, delta, H)
        build_sb_constraints(solver, lp_instance, delta, x, y, H)
    if method == "rotations":
        x, y, r = build_rotate_constraints(solver, lp_instance, delta, H)
    if method == "rotations-sb":
        x, y, r = build_rotate_constraints(solver, lp_instance, delta, H)
        build_rotations_sb_constraints(solver, lp_instance, r)

    # solve
    if verbose:
        print("Solving process started")
    solver.set_time_limit(timeout*(10**3))
    solver.Minimize(H)

    status = solver.Solve(solver_parameters)

    solve_time = solver.wall_time() / 10**3

    time_over = None
    solution_found = None

    if status == pywraplp.Solver.NOT_SOLVED:
        time_over = True
        solution_found = "UPPER_BOUND"
        solve_time = timeout+1
        if verbose:
            print(f"instance {index} not solved")

    elif not(status == pywraplp.Solver.NOT_SOLVED):
        # case in which the solver is gone overtime
        if not(status == pywraplp.Solver.OPTIMAL) or (solve_time >= timeout):
            solution_found = "NOT_OPTIMAL"
            time_over = True
            solve_time = timeout+1
            if verbose:
                print(f"instance {index} solved, but solution is not optimal")

        elif status == pywraplp.Solver.OPTIMAL:
            solution_found = "OPTIMAL"
            time_over = False
            if verbose:
                print(f"Instance {index} solved in ", solve_time)

        if method == "rotations" or method == "rotations-sb":
            for i in range(0, lp_instance.n_instances):
                if int(r[i].solution_value()) == 1:
                    lp_instance.rectangles[i].width = height[i]
                    lp_instance.rectangles[i].height = width[i]

        for i in range(0, lp_instance.n_instances):
            lp_instance.rectangles[i].x = int(x[i].solution_value())
            lp_instance.rectangles[i].y = int(y[i].solution_value())

        lp_instance.H = int(H.solution_value())

        if plot:
            # plot results
            plot_rectangles(lp_instance.rectangles, url)

    path = method + "/" + solver_name
    write_log(path="../out/lp/" + path + "/" + file, instance=lp_instance,
              add_text="\n" + str(solve_time) + "\n" + str(time_over) + "\n" + solution_found)

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


#for i in range(1,40):
#    lp_benchmark(i, 300, "base-sb", "gurobi", verbose = False, plot=True)

# timeout is set in seconds
plot_LP_benchmark(instances_to_solve=40, solver_name="scip", timeout=300, plot=True)

