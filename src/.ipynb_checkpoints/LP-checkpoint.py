%load_ext iminizinc
import sys
sys.path.append('../')
from utils.utils import *
from minizinc import Instance, Model, Solver

plot_rectangles([Rectangle(21,6,4,8), Rectangle(15,8,63,4)])

import asyncio
loop = asyncio.get_event_loop()
# Load n-Queens model from file
model = Model("./LP.mzn")
# Find the MiniZinc solver configuration for Gecode
solver = Solver.lookup("gurobi")
# Create an Instance of the n-Queens model for Gecode
instance = Instance(solver, model)
# Assign 4 to n
instance["h_ub"] = 4
instance["h_lb"] = 4
instance["W"] = 4
instance["n_rectangles"] = 4
instance["pos_x"] = 4
instance["pos_y"] = 4


result = instance.solve_async()
print(result)

# Output the array q
#print(result["q"])

