# 2D Strip Packing Problem - VLSI
In this project, we try 4 different optimization techniques to solve the very well known problem of 2D Strip Packing. To model it in an intuitive way.
To understand the reasons of our implementations, read the ``Report.pdf`` included in this repo.

©Alessandro D’Amico ©Andrea Virgillito ©Sfarzo El Husseini



## Requirements
Some general requirements all valid for all the methods, since they use common python 
packages for the upper bound computation, plotting and parsing of the iinput/output instances:

- python (tested on Python 3.8)
- pandas ``pip install pandas``
- tqdm ``pip install tqdm``
- matplotlib ``pip install matplotlib``
- numpy ``pip install numpy``
- re ``pip install regex``
- gc
- datetime ``pip install DateTime``
- time 


The requirements are different across the various modeling techniques:

- cp: 
	- Minizinc install & IDE (tested on version 2.5.5)
	- OrTools flatzinc install (tested on the visual studio 2019 64 bit v 9.2.9972) 
	  with all the flags checked through the Minizinc IDE (make sure to properly link it)
	- Minizinc Python API (can easily install it with pip: ``pip install minizinc``)

- sat & smt:
	- Z3 Python API (can easily install it with pip: ``pip install z3-solver``)
- lp:
	- OrTools Python API (can easily install it with pip: ``pip install ortools``).
	- Gurobi install (in our case activated with academic license, which can be obtained
	  just through the university network) (tested on version 9.5.2)
	- recompilation of the whole C++ package (which is failing right now 03/12/2022 )
	  is required if one needs to try other commercial solvers such as CPLEX 
	  (of course, after having installed them on the computer in use and having set a 
	  system path variable for em) - more info on 
	  https://developers.google.com/optimization/install/python
	
## PLAY WITH IT :)
When the corresponding enviroment is set as above,

- cp:
	- run ``python CP.py`` in the cp folder to process the whole benchmark on Or-Tools or use the functions
	  inside that file to process just a single instance.
- sat:
 	- run ``python SAT.py`` in the sat folder to process the whole benchmark on Z3 or use the functions
	  inside that file to process just a single instance.
- smt:
 	- run the notebook ``smt.ipynb`` in the smt/src folder to process the whole benchmark on Z3 or use the functions
	  inside that file to process just a single instance.
- lp:
 	- run ``python LP.py`` in the lp folder to process the whole benchmark on Gurobi or use the functions
	  inside that file to process just a single instance.




## Input format

The collections of rectangles to be placed are located in the 
"instances" folder.
Each of them is named "ins-{x}.txt", where {x} is the instance number.
Each one of them is of this form:

```
_____________ ins-{x}.txt _____________  
 W  
 n_rect  
 width(0) height(0)  
 width(1) height(1)  
 ...  
 width(n_rect) height(n_rect)  
 _____________________________________
```
where we have

- ``W``: the width of the strip
- ``n_rect``: the number of rectangles to be placed in the corresponding instance (collection)
- ``width(i)``: the width of the rectangle i
- ``height(i)``: the height of the rectangle i

 _____________________________________

## Output format

To visualize singular solutions use the notebook 'visualize_solutions.ipynb'

The computed solutions (the placements of the rectangles) are placed in 
the "out" folder, divided by 
- modeling technique: cp, sat, smt, lp
- strategy: 
	- base (base version of the problem, no rotations), 
	- base-sb (base version with the addition of symmetry breaking 
	  techniques)
	- rotations (model allowing the rotation of rectangles)
	- rotations-sb (model allowing rotations, with the addition of 
	  symmetry breaking techniques)
- solver (depending on the modeling technique, sometimes different solvers
	  were used)

```
_____________ ins-{x}.txt _____________  
 W H  
 n_rectangles  
 width(0) height(0) x_(0) y_(0) rot(0)  
 width(1) height(1) x_(1) y_(1) rot(1)  
 ...  
 width(n_rect) height(n_rect) x(n_rect) y(n_rect) rot(n_rect)  
   
 time  
 overtime  
 kind_of_bound  
 _____________________________________
```
In these output files there are some added numbers (wrt the plain input file described above)

- ``H``: the overall height of the strip
- ``x(i)``: the position on the x-axis of the low bottom corner of the 
	   rectangle i 
- ``y(i)``: the position on the x-axis of the low bottom corner of the 
	   rectangle i
- ``rot(i)``: "rot" if the rectangle i has been rotated (WARNING: in that 
	    case the height and the width are already swapped),
	    "not-rot" otherwise
- ``time``: the time taken from the solver (in seconds)
- ``overtime``: True if the solver has exceeded the time limit (300s in our tests)
- kind_of_bound: 
	- ``OPTIMAL`` if the height corresponding to the perfect packaging has been reached
	- ``NOT_OPTIMAL`` if at least a valid solution has been computed by the solver within the time limit
	- ``UPPER_BOUND`` if no solution from the solver has been computed within the time limit:
	  in this case, we use just the height (and the positioning of the rectangles)
	  computed by the method used to obtain the upper bound (which is )    
 _____________________________________
