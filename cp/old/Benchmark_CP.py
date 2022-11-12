import numpy as np 
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm
# %matplotlib inline


times_base = [0.015, 0.010, 0.015, 0.020, 0.021, 0.028, 0.014,  0.023,  0.028, 0.021,
0.072, 0.044, 0.033, 0.072, 0.041, 0.732, 0.118, 0.234, 0.303, 0.517,
3.014, 1.083, 0.264, 0.602, 137.962, 0.493, 0.324, 0.381, 2.898, 6.697,
0.704, 194.393, 0.230, 1.419, 1.363,  0.156, 42.091, 22.323, 2.866, 300]
times_base_rotate = [0.018,  0.022,   0.025, 0.019, 0.054, 0.040, 0.093, 0.051, 0.045,  0.093,
2.262,  0.498,   0.092, 0.309, 0.385, 0.901, 2.073, 4.337, 23.648, 5.175,
35.544, 218.048, 8.944, 1.338,   300,   7.094, 3.730, 8.480, 12.201,   300,
0.435,    300,     2.452, 1.310, 0.531, 0.795,   300,   6.451,   300,      300]
time_overs_base = [40]
time_overs_base_rotate = [25,30,32, 37, 39, 40]

print(times_base[33])
quit()

def plot_CP_benchmark(times_base, times_base_rotate):
  '''
  Plotting the barplot with all the CP solving mechanisms (base, rotated)
  
  '''

  X = range(1, (len(times_base) + 1)*4, 4) 
  X_axis = np.arange(1,len(times_base)+1)
  
  plt.rcParams["figure.figsize"] = (10,5.5)
    
    
  # base 
  barbase = plt.bar(X_axis - 0.2, times_base, 0.4, label = 'Base time')
  

  for i in time_overs_base:  
    barbase[i-1].set_alpha(0.25)
  
  # base + rotation 
  barbaserotation = plt.bar(X_axis + 0.2, times_base_rotate, 0.4, label = 'Rotated time')
  
  for i in time_overs_base_rotate:  
    barbaserotation[i-1].set_alpha(0.25)
    
  plt.xlabel("Instance files")
  plt.ylabel("Time(s)")
  plt.title("VLSI CP Benchmark")
  plt.grid()
  #plt.axhline(y=timeout, xmin=0.1, xmax=0.8, color='r', linestyle='-.', linewidth=2, label=f"time_limit = {timeout} s")
  plt.yscale("log")
  plt.legend()
  plt.show()

  print(f"total Base time        -- mean: {np.mean(times_base)} std: {np.std(times_base)}")
  print(f"total rotated time     -- mean: {np.mean(times_base_rotate)} std: {np.std(times_base_rotate)}")

#timeout is set in seconds
plot_CP_benchmark(times_base, times_base_rotate)
