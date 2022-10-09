% input parameters
int: h_ub;
int: h_lb;
int: W;
int: n_rectangles;
int: biggest_rect;

% ranges for loops
set of int: RECTANGLES = 1..n_rectangles;
 
% input arrays of rectangles dimensions
array[RECTANGLES] of int: rect_height; 
array[RECTANGLES] of int: rect_width; 

% output placing coordinates
array[RECTANGLES] of var int: rect_x;
array[RECTANGLES] of var int: rect_y; 

% variables to encode "OR" condition
array[RECTANGLES, RECTANGLES, 1..2] of var 0..1: delta; 

%constraint forall(i in RECTANGLES)(delta[i,i,1] = 1);
%constraint forall(i in RECTANGLES)(delta[i,i,2] = 1);
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

% redundant constraints
%constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,1] + delta[j,i,1] >= 1);
%constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,2] + delta[j,i,2] >= 1);

% other constraints
%constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,1] + delta[i,j,2] + delta[j,i,1] + delta[j,i,2] >= 1);
% symmetry breaking
%constraint rect_x[biggest_rect] = 0;
%constraint rect_y[biggest_rect] = 0;


solve minimize H;