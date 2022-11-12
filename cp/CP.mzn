include "globals.mzn";

% Loading the instance
int: n_components;   % Number of components
int: fixed_width; % Fixed width
int: h_ub;
set of int: COMPONENTS = 1..n_components;   % or anon_enum(n_components);
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
%-----------------------------------

solve :: seq_search([
int_search(y_coords, first_fail, indomain_min),
int_search(x_coords, first_fail, indomain_min),
int_search([height], smallest, indomain_min)])
minimize height;