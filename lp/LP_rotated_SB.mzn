% input parameters
int: h_ub;
int: h_lb;
int: W;
int: n_rectangles;
int: n_squares;

% ranges for loops
set of int: RECTANGLES = 1..n_rectangles;
set of int: SQUARES = 1..n_squares;
 
% input arrays of rectangles dimensions
array[RECTANGLES] of int: rect_height; 
array[RECTANGLES] of int: rect_width; 

% indexes of squares
array[SQUARES] of int: square_index;

% output placing coordinates
array[RECTANGLES] of var int: rect_x;
array[RECTANGLES] of var int: rect_y; 

% variables to encode "OR" no-overlap condition 
array[RECTANGLES, RECTANGLES, 1..2] of var 0..1: delta; 

% variables to model possible rotations
array[RECTANGLES] of var 0..1: rotated;
array[RECTANGLES] of var 0..W: rect_new_width;
array[RECTANGLES] of var 0..h_ub: rect_new_height;

%height bounds setting
var h_lb..h_ub: H;

% constraint that ensures that rectangles are inside the margins W and H
constraint forall(i in RECTANGLES)( rect_x[i] + rect_new_width[i] <= W);
constraint forall(i in RECTANGLES)( rect_y[i] + rect_new_height[i] <= H);
constraint forall(i in RECTANGLES)( rect_y[i] >= 0);
constraint forall(i in RECTANGLES)( rect_x[i] >= 0);

% constraints for non-overlapping
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_x[i] + rect_new_width[i] - rect_x[j] - (W*delta[i,j,1]) <= 0);
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_x[j] + rect_new_width[j] - rect_x[i] - (W*delta[j,i,1]) <= 0);
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_y[i] + rect_new_height[i] - rect_y[j] - (H*delta[i,j,2]) <= 0);
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_y[j] + rect_new_height[j] - rect_y[i] - (H*delta[j,i,2]) <= 0);

% constraints that ensures that at least one of the previous condition (in a logical OR) is satisfied
constraint forall(i in RECTANGLES, j in 1..i-1)(delta[i,j,1] + delta[i,j,2] + delta[j,i,1] + delta[j,i,2] <= 3);

% constraints to define the possibility of rotations
constraint forall(i in RECTANGLES)(rect_new_width[i] = rotated[i]*rect_height[i] + (1 - rotated[i])*rect_width[i]);
constraint forall(i in RECTANGLES)(rect_new_height[i] = rotated[i]*rect_width[i] + (1 - rotated[i])*rect_height[i]); 

% symmetry breaking to avoid rotation of squares
constraint forall(i in SQUARES)(rotated[square_index[i]] = 0);

solve minimize H;