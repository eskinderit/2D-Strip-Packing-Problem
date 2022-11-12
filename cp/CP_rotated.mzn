include "globals.mzn";

% Loading instances
int: h_ub;
int: n_components;   % Number of components
int: fixed_width; % Fixed width
int: h_lb = (sum(k in COMPONENTS)(widths[k] * heights[k])) div fixed_width;
int: t=max(max(widths), max(heights));
set of int: COMPONENTS = 1..n_components;   % or anon_enum(n_components);

array[COMPONENTS] of int: widths; % components width(s)
array[COMPONENTS] of int: heights; % components height(s)
array[COMPONENTS] of var 0..t:actual_widths;
array[COMPONENTS] of var 0..t:actual_heights;
array[COMPONENTS] of var bool: rot;

var h_lb..h_ub: height;

% Defining variables

array[COMPONENTS] of var 0..fixed_width-min(min(heights),min(widths)): x_coords; % lower left corner x coordinates
array[COMPONENTS] of var 0..2*max(h_lb,max(heights))-min(min(heights),min(widths)): y_coords;


%-----------------------------------CONSTRAINTS
constraint diffn(x_coords, y_coords, actual_widths, actual_heights);
constraint forall(i in 1..n_components)(x_coords[i] + actual_widths[i]<=fixed_width);
constraint forall(i in 1..n_components)(y_coords[i] + actual_heights[i]<=height);
constraint cumulative(y_coords, actual_heights, [actual_widths[j] | j in COMPONENTS], fixed_width);
constraint forall(i in COMPONENTS)
  ((rot[i] /\ actual_widths[i] == heights[i] /\ actual_heights[i] == widths[i])
  xor
  (not(rot[i]) /\ actual_widths[i] == widths[i] /\ actual_heights[i] == heights[i]));
%-----------------------------------

solve :: seq_search([
int_search(y_coords, first_fail, indomain_min),
int_search(x_coords, first_fail, indomain_min),
int_search([height], smallest, indomain_min),
int_search(rot, smallest, indomain_min, complete)])
minimize height;