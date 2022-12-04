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
array[COMPONENTS] of var 0..h_ub-min(heights): y_coords; % lower left corner y coordinates


%-----------------------------------CONSTRAINTS
constraint diffn(x_coords, y_coords, widths, heights);
constraint cumulative(y_coords, heights, widths, fixed_width);
constraint forall(i in 1..n_components)((x_coords[i] + widths[i])<=fixed_width);
constraint forall(i in 1..n_components)((y_coords[i] + heights[i])<=height);
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