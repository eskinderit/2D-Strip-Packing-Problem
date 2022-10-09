% input parameters
int: h_ub;
int: h_lb;
int: W;
int: n_rectangles;
int: biggest_rect;
int: n_large_rectangles;
int: n_same_dim_rectangles;
int: n_smaller_rectangles;

% ranges for loops
set of int: RECTANGLES = 1..n_rectangles;
set of int: LARGE_RECTANGLES = 1..n_large_rectangles;
set of int: SAME_DIM_RECTANGLES = 1..n_same_dim_rectangles;
set of int: SMALLER_RECTANGLES = 1..n_smaller_rectangles;
 
array[LARGE_RECTANGLES,1..2] of int: large_rectangles;
array[SAME_DIM_RECTANGLES,1..2] of int: same_dim_rectangles;
 
% input arrays of rectangles dimensions
array[RECTANGLES] of int: rect_height; 
array[RECTANGLES] of int: rect_width; 

% output placing coordinates
array[RECTANGLES] of var int: rect_x;
array[RECTANGLES] of var int: rect_y; 

% variables to encode "OR" condition
array[RECTANGLES, RECTANGLES, 1..2] of var 0..1: delta; 

%height bounds setting
var h_lb..h_ub: H;

% constraint that ensures that rectangles are inside the margins W and H
constraint forall(i in RECTANGLES)( rect_x[i] + rect_width[i] <= W);
constraint forall(i in RECTANGLES)( rect_y[i] + rect_height[i] <= H);
constraint forall(i in RECTANGLES)( rect_y[i] >= 0);
constraint forall(i in RECTANGLES)( rect_x[i] >= 0);

% constraints for non-overlapping
constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_x[i] + rect_width[i] - rect_x[j] - (W*delta[i,j,1]) <= 0);
constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_x[j] + rect_width[j] - rect_x[i] - (W*delta[j,i,1]) <= 0);
constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_y[i] + rect_height[i] - rect_y[j] - (H*delta[i,j,2]) <= 0);
constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_y[j] + rect_height[j] - rect_y[i] - (H*delta[j,i,2]) <= 0);

% sum delta[i,j,k]=1 on k in [1,4]
constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,1] + delta[i,j,2] + delta[j,i,1] + delta[j,i,2] <= 3);

% symmetry breaking for diagonal elements of dummy variables
constraint forall(i in RECTANGLES)(delta[i,i,1] = 1);
constraint forall(i in RECTANGLES)(delta[i,i,2] = 1);

% symmetry breaking for rectangles that are such that width[i]+width[j] > W
constraint forall(i in LARGE_RECTANGLES)( delta[large_rectangles[i,1],large_rectangles[i,2],1] = 1 );
constraint forall(i in LARGE_RECTANGLES)( delta[large_rectangles[i,2],large_rectangles[i,1],1] = 1 );

% symmetry breaking constraint for same sized rectangles. We make sure that the first rectangle is left and/or down the second rectangle
% i can't be above j
constraint forall(i in SAME_DIM_RECTANGLES)( delta[same_dim_rectangles[i,2],same_dim_rectangles[i,1],1] = 1 );
constraint forall(i in SAME_DIM_RECTANGLES)( delta[same_dim_rectangles[i,2],same_dim_rectangles[i,1],2] = 1 );
% i can't be right to j

% symmetry breaking for biggest rectangle
constraint 2*rect_x[biggest_rect] <= W - rect_width[biggest_rect];
constraint 2*rect_y[biggest_rect] <= H - rect_height[biggest_rect];
constraint forall(i in SMALLER_RECTANGLES)(delta[i,biggest_rect,1] = 1);

solve minimize H;