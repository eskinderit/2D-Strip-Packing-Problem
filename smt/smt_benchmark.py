def plot_SMT_benchmark(instances_to_solve: int = 40, solver_name: str = "gurobi", timeout: int = 300, plot=False):
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
        time, time_over = #FIXME smt_benchmark(j, timeout=timeout, method="base", solver_name=solver_name, plot=plot)
        times_base.append(time)

        if time_over:
            time_overs_base.append(j - 1)

        # base + SB
        time, time_over = #FIXME smt_benchmark(j, timeout=timeout, method="base-sb", solver_name=solver_name, plot=plot)
        times_SB.append(time)

        if time_over:
            time_overs_SB.append(j - 1)

        # rotated
        time, time_over = #FIXME smt_benchmark(j, timeout=timeout, method="rotations", solver_name=solver_name, plot=plot)
        times_base_rotate.append(time)

        if time_over:
            time_overs_base_rotate.append(j - 1)

        # rotated + SB
        time, time_over = #FIXME smt_benchmark(j, timeout=timeout, method="rotations-sb", solver_name=solver_name, plot=plot)
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
    plt.title("VLSI SMT Benchmark" + "solver: " + solver_name)
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
    with open(f"smt_benchmark_log_{solver_name}.txt", "w") as file:
        content = out_text
        file.writelines(content)
        file.close()

