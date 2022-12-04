include "globals.mzn";

% Loading the instance
int: n_components;   % number of components
int: fixed_width; % width of the silicon plate
int: h_ub;
set of int: COMPONENTS = 1..n_components;   % or anon_enum(n_components);
array[COMPONENTS] of int: widths; % components width(s)
array[COMPONENTS] of int: heights; % components height(s)
int: biggest_rect = arg_max([widths[i]*widths[i] | i in 1..n_components]); % index of the rectangle having the biggest area
set of int: SMALLER = COMPONENTS diff {biggest_rect};

% Defining variables

int: h_lb = (sum(k in COMPONENTS)(widths[k] * heights[k])) div fixed_width;
var h_lb..h_ub: height;

array[COMPONENTS] of var 0..fixed_width-min(widths): x_coords; % lower left corner x coordinates
array[COMPONENTS] of var 0..h_ub-min(heights): y_coords; % lower left corner y coordinates


%-----------------------------------CONSTRAINTS
constraint diffn(x_coords, y_coords, widths, heights);
constraint cumulative(y_coords, heights, widths, fixed_width);
constraint forall(i in 1..n_components)((x_coords[i] + widths[i])<=fixed_width);
constraint forall(i in 1..n_components)((y_coords[i] + heights[i])<=height);

%----------------------------------- SYMMETRY BREAKING CONSTRAINTS

% LS - symmetry breaking for biggest rectangle

constraint symmetry_breaking_constraint(x_coords[biggest_rect] <= (fixed_width - widths[biggest_rect]) div 2 /\ 
                                        y_coords[biggest_rect] <= (height - heights[biggest_rect]) div 2 );


constraint redundant_constraint(forall(i in SMALLER)(if (widths[i] > (fixed_width - widths[biggest_rect]) div 2) then
                                                              not(x_coords[i] < x_coords[biggest_rect] /\
                                                                  y_coords[i] < y_coords[biggest_rect]+heights[biggest_rect] /\
                                                                  y_coords[i] + heights[i]>= y_coords[biggest_rect]
                                                                  ) endif));
                 

% SR - symmetry breaking for items of identical dimensions

constraint symmetry_breaking_constraint(forall(i,j in COMPONENTS where j>i)
                                            (if widths [i]== widths [j] /\ heights [i]==heights [j] 
                                            then x_coords[i] <= x_coords[j] /\ y_coords[i] <= y_coords[j] endif));
/*
% LR - symmetry breaking for rectangles such that width[i] + width[j] > W

constraint redundant_constraint(forall(i,j in COMPONENTS where j>i)
                                            (if widths[i] + widths[j] > fixed_width
                                            then abs(y_coords[i] - y_coords[j]) > min(heights[i],heights[j]) endif));
*/
%-----------------------------------

array [1..2* n_components ] of var int: coords_couples = [if ( i mod 2 = 0) then
      x_coords [ (i div 2)] else y_coords[((i+1) div 2)] endif | i in 1..2* n_components ];

solve :: seq_search([
% we first try increasing values for height
int_search([height], input_order, indomain_min),
% in sequence, we try (x,y) coordinates choosing y and then x of the corresponding rectangle
int_search (coords_couples , input_order , indomain_min)
])

minimize height;