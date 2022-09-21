import sys
from minizinc import Instance, Model, Solver
sys.path.append('../')
from utils.utils import *


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
instance["rect_length"] = [7,5,6,8]
instance["rect_width"] = [9,2,4,9]


result = instance.solve()
print(result.solution.rect_x)

# Output the array q
#print(result["q"])

