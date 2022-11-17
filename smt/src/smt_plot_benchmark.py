import numpy as np 
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm
%matplotlib inline

def plot_SMT_benchmark(instances_to_solve = 5,timeout = 300, plot=False):
    '''
    Plotting the barplot with all the SMT solving mechanisms (base, rotated, 
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
        _, _, timer, time_over, z3_timer = #FIXME VLSI_SMT_solver().solve(instance_path = url, timeout= timeout, break_symmetries = True ,rotate=False, verbose = plot, plot = plot)
        times_SB.append(timer)
        z3_times_SB.append(z3_timer)

        if(time_over):
            time_overs_SB.append(i-1)

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = #FIXME VLSI_SMT_solver().solve(instance_path = url, timeout= timeout, break_symmetries = True ,rotate=True, verbose = plot, plot = plot)
        times_SB_rotate.append(timer)
        z3_times_SB_rotate.append(z3_timer)

        if(time_over):
            time_overs_SB_rotate.append(i-1)

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = #FIXME VLSI_SMT_solver().solve(instance_path = url, timeout= timeout, break_symmetries = False ,rotate=False, verbose = plot, plot = plot)
        times_base.append(timer)
        z3_times_base.append(z3_timer)

        if(time_over):
            time_overs_base.append(i-1)

        url = f"../instances/ins-{i}.txt"
        _, _, timer, time_over, z3_timer = #FIXME VLSI_SMT_solver().solve(instance_path = url, timeout= timeout, break_symmetries = False ,rotate=True, verbose = plot, plot = plot)
        times_base_rotate.append(timer)
        z3_times_base_rotate.append(z3_timer)

        if(time_over):
            time_overs_base_rotate.append(i-1)

    X = range(1, instances_to_solve + 1) 
    X_axis = np.arange(0,len(times_SB)*2,2)


    plt.rcParams["figure.figsize"] = (13,6)
    plt.xticks(X_axis, X)



    # base z3
    barbase = plt.bar(X_axis - 0.6, z3_times_base, 0.4, label = 'Base z3 time')
  
    # base total
    plt.plot(X_axis, times_base, linestyle='--', marker='o', label='Base total time')

    for i in time_overs_base:  
        barbase[i].set_alpha(0.25)
  
    # base + rotation z3
    barbaserotation = plt.bar(X_axis - 0.2, z3_times_base_rotate, 0.4, label = 'Rotated z3 time')
  
    # base + rotation total
    plt.plot(X_axis, times_base_rotate, linestyle='--', marker='o', label='Rotated total time')

    for i in time_overs_base_rotate:
        barbaserotation[i].set_alpha(0.25)

    # SB
    barSB = plt.bar(X_axis + 0.2, z3_times_SB, 0.4, label = 'SB z3 time')
  
    # SB total
    plt.plot(X_axis, times_SB, linestyle='--', marker='o', label='SB total time')

    for i in time_overs_SB:  
        barSB[i].set_alpha(0.25)
        
    # SB + rotation
    barSBrotation = plt.bar(X_axis + 0.6, z3_times_SB_rotate, 0.4, label = 'Rotated + SB z3 time')

    #SB + rotation total
    plt.plot(X_axis, times_SB_rotate, linestyle='--', marker='o', label='Rotated + SB total time')

    for i in time_overs_SB_rotate:  
        barSBrotation[i].set_alpha(0.25)

    plt.xlabel("VLSI instance files")
    plt.ylabel("Time(s)")
    plt.title("VLSI SMT Benchmark")
    plt.grid()
    plt.axhline(y=timeout, xmin=0, xmax=1, color='r', linestyle='-.', linewidth=2, label=f"time_limit = {timeout} s")
    #plt.yscale("log")
    plt.legend()
    plt.savefig('smt_benchmark.png', transparent=False,format="png")
    plt.show()

    out_text= ""

    out_text += f"total Base time        -- mean: {np.mean(times_base)} std: {np.std(times_base)} \n"
    out_text += f"total rotated time     -- mean: {np.mean(times_base_rotate)} std: {np.std(times_base_rotate)} \n"
    out_text += f"total SB time          -- mean: {np.mean(times_SB)} std: {np.std(times_SB)} \n"
    out_text += f"total rotated + SB time-- mean: {np.mean(times_SB_rotate)} std: {np.std(times_SB_rotate)} \n"

    out_text += f"z3 Base time           -- mean: {np.mean(z3_times_base)} std: {np.std(z3_times_base)} \n"
    out_text += f"z3 rotated time        -- mean: {np.mean(z3_times_base_rotate)} std: {np.std(z3_times_base_rotate)} \n"
    out_text += f"z3 SB time             -- mean: {np.mean(z3_times_SB)} std: {np.std(z3_times_SB)} \n"
    out_text += f"z3 rotated + SB time   -- mean: {np.mean(z3_times_SB_rotate)} std: {np.std(z3_times_SB_rotate)}"

 # write txt log
    with  open("smt_benchmark_log.txt","w") as file:
        content = out_text
        file.writelines(content)
        file.close()
        
    print(out_text)
#timeout is set in seconds
plot_SMT_benchmark(instances_to_solve=40, timeout=300, plot=False)