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
var h_lb..h_ub: lt;

% constraint that ensures that rectangles are inside the margins W and H
%constraint forall(i in RECTANGLES)( rect_x[i] + rect_width[i] <= W);
%constraint forall(i in RECTANGLES)( rect_y[i] + rect_height[i] <= H);
%constraint forall(i in RECTANGLES)( rect_y[i] >= 0);
%constraint forall(i in RECTANGLES)( rect_x[i] >= 0);

constraint forall(i in 1..n_rectangles)( lt >= rect_x[i]+rect_width[i]);

% constraints for non-overlapping
constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_x[i] + rect_width[i] <= rect_x[j] + (h_ub*delta[j,i,1]) + 
  (rect_width[i] + rect_width[j])*(delta[i,j,2] + delta[j,i,2]));

constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_x[j] + rect_width[j] <= rect_x[i] + (h_ub*delta[i,j,1]) +
  (rect_width[i] + rect_width[j])*(delta[i,j,2] + delta[j,i,2]));
  
constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_y[i] - rect_height[i] >= rect_y[j] - W*(delta[i,j,1]+delta[j,i,1]+delta[j,i,2]));

constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_x[i] + rect_width[i] >= rect_x[j] - (h_ub - rect_width[i] - rect_width[j])*delta[i,j,1] - (rect_width[j] - rect_width[i])*delta[j,i,1]);

constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_x[j] + rect_width[j] >= rect_x[i] - (h_ub - rect_width[i] - 
rect_width[j])*delta[j,i,1] - (rect_width[i] - rect_width[j])*delta[i,j,1]);

constraint forall(i in 1..n_rectangles, j in 1..i-1)(rect_y[j] - rect_height[j] >= rect_y[i] - W*(delta[i,j,1] +delta[j,i,1]+delta[i,j,2]));

% sum delta[i,j,k]=1 on k in [1,4]
constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,1] + delta[i,j,2] + delta[j,i,1] + delta[j,i,2] = 1);


constraint forall(i in 1..n_rectangles)(0<= rect_x[i]);
constraint forall(i in 1..n_rectangles)(rect_x[i]<= (h_ub-rect_width[i]));
constraint forall(i in 1..n_rectangles)(rect_height[i]<= rect_y[i]);
constraint forall(i in 1..n_rectangles)(rect_y[i]<= W);

% redundant constraints
%constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,1] + delta[j,i,1] >= 1);
%constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,2] + delta[j,i,2] >= 1);

% other constraints
%constraint forall(i in 1..n_rectangles, j in 1..i-1)(delta[i,j,1] + delta[i,j,2] + delta[j,i,1] + delta[j,i,2] >= 1);

% symmetry breaking
constraint 2*rect_x[biggest_rect] <= lt - rect_width[biggest_rect];
constraint 2*rect_y[biggest_rect] >= W - rect_height[biggest_rect];


solve minimize lt;