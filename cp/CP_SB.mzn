include "globals.mzn";

% Loading the instance
int: n_components;   % Number of components
int: fixed_width; % Fixed width
int: h_ub;
int: biggest_rect;
set of int: COMPONENTS = 1..n_components;   % or anon_enum(n_components);
set of int: SMALLER = COMPONENTS diff {biggest_rect};
array[COMPONENTS] of int: widths; % components width(s)
array[COMPONENTS] of int: heights; % components height(s)

% Defining variables

int: h_lb = (sum(k in COMPONENTS)(widths[k] * heights[k])) div fixed_width;
var h_lb..h_ub: height;

array[COMPONENTS] of var 0..fixed_width-min(widths): x_coords; % lower left corner x coordinates
array[COMPONENTS] of var 0..2*max(h_lb,max(heights))-min(heights): y_coords; % lower left corner y coordinates


%-----------------------------------CONSTRAINTS
constraint diffn(x_coords, y_coords, widths, heights);
constraint cumulative(y_coords, heights, [widths[j] | j in COMPONENTS], fixed_width);
constraint max(i in 1..n_components)(x_coords[i] + widths[i])<=fixed_width;
constraint max(i in 1..n_components)(y_coords[i] + heights[i])<=height;

%----------------------------------- SYMMETRY BREAKING CONSTRAINTS
% LS - symmetry breaking for biggest rectangle
constraint symmetry_breaking_constraint(x_coords[biggest_rect] <= (fixed_width - widths[biggest_rect]) div 2 /\ 
                                        y_coords[biggest_rect] <= (height - heights[biggest_rect]) div 2 );

constraint redundant_constraint(forall(i in SMALLER)(if (widths[i] > (fixed_width - widths[biggest_rect] div 2)) then 
                                        abs(y_coords[i] - y_coords[biggest_rect]) >= min(heights[i], heights[biggest_rect])endif));
                                        
% SR - symmetry breaking for items of identical dimensions
constraint symmetry_breaking_constraint(forall(i,j in COMPONENTS where j>i)
                                            (if widths [i]== widths [j] /\ heights [i]==heights [j] 
                                            then x_coords[i] <= x_coords[j] /\ y_coords[i] <= y_coords[j] endif));
% L - symmetry breaking for rectangles such that width[i] + width[j] > W
constraint redundant_constraint(forall(i,j in COMPONENTS where j>i)
                                            (if widths[i] + widths[j] > fixed_width
                                            then abs(y_coords[i] - y_coords[j]) > min(heights[i],heights[j]) endif));
%-----------------------------------

solve :: seq_search([
int_search(y_coords, first_fail, indomain_min),
int_search(x_coords, first_fail, indomain_min),
int_search([height], smallest, indomain_min)])
minimize height;