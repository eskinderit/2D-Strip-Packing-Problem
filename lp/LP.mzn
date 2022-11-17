% input parameters
int: h_ub;
int: h_lb;
int: W;
int: n_rectangles;
var int: H;

% ranges for loops
set of int: RECTANGLES = 1..n_rectangles;
 
% input arrays of rectangles dimensions
array[RECTANGLES] of int: rect_height; 
array[RECTANGLES] of int: rect_width; 

% output placing coordinates
array[RECTANGLES] of var int: rect_x;
array[RECTANGLES] of var int: rect_y; 

% variables to encode "OR" condition
array[RECTANGLES, RECTANGLES, 1..2] of var int: delta; 
constraint forall(i in RECTANGLES,j in RECTANGLES,k in 1..2)(0 <= delta[i,j,k]);
constraint forall(i in RECTANGLES,j in RECTANGLES,k in 1..2)(delta[i,j,k] <= 1);

% constraint that ensures that rectangles are inside the margins W and H
constraint forall(i in RECTANGLES)( rect_x[i] + rect_width[i] <= W);
constraint forall(i in RECTANGLES)( rect_y[i] + rect_height[i] <= H);
constraint forall(i in RECTANGLES)( rect_y[i] >= 0);
constraint forall(i in RECTANGLES)( rect_x[i] >= 0);

% constraints for non-overlapping
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_x[i] + rect_width[i] <= rect_x[j] + (W*delta[i,j,1]) );
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_x[j] + rect_width[j] <= rect_x[i] + (W*delta[j,i,1]) );
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_y[i] + rect_height[i] <= rect_y[j] + (h_ub*delta[i,j,2]) );
constraint forall(i in RECTANGLES, j in 1..i-1)(rect_y[j] + rect_height[j] <= rect_y[i] + (h_ub*delta[j,i,2]) );

% sum delta[i,j,k]=1 on k in [1,4]
constraint forall(i in RECTANGLES, j in 1..i-1)(delta[i,j,1] + delta[i,j,2] + delta[j,i,1] + delta[j,i,2] <= 3);

%height bounds setting
constraint H >= h_lb;
constraint H <= h_ub;

solve minimize H;