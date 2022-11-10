include "globals.mzn";

% Loading instances
int: n_components;   % Number of components
set of int: COMPONENTS = 1..n_components;   % or anon_enum(n_components);
array[COMPONENTS] of int: widths; % components width(s)
int: fixed_width; % Fixed width
array[COMPONENTS] of int: heights; % components height(s)

int: t=max(max(widths), max(heights));
array[COMPONENTS] of var 0..t:actual_widths;
array[COMPONENTS] of var 0..t:actual_heights;
array[COMPONENTS] of var bool: rot;
constraint forall(i in COMPONENTS)
((rot[i] /\ actual_widths[i] == heights[i] /\
actual_heights[i] == widths[i])
xor
(not(rot[i]) /\ actual_widths[i] == widths[i]
/\ actual_heights[i] == heights[i]));

% Defining variables

array[COMPONENTS] of var 0..fixed_width-min(min(heights),min(widths)): x_coords; % lower left corner x coordinates
array[COMPONENTS] of var 0..2*max(min_h,max(heights))-min(min(heights),min(widths)): y_coords;

int: min_h = (sum(k in COMPONENTS)(widths[k] * heights[k])) div fixed_width;

var min_h..2*max(min_h,max(heights)): height;

%-----------------------------------CONSTRAINTS
constraint diffn(x_coords, y_coords, actual_widths, actual_heights);
constraint forall(i in 1..n_components)(x_coords[i] + actual_widths[i]<=fixed_width);
constraint forall(i in 1..n_components)(y_coords[i] + actual_heights[i]<=height);
constraint cumulative(y_coords, actual_heights, [actual_widths[j] | j in COMPONENTS], fixed_width);
constraint symmetry_breaking_constraint(
  lex_lesseq([x_coords[k] | k in COMPONENTS], [fixed_width - x_coords[k] - actual_widths[k] | k in COMPONENTS]));
%-----------------------------------


%-----------------------------------SYMMETRY BREAKING CONSTRAINTS
constraint symmetry_breaking_constraint(lex_lesseq([x_coords[k] | k in COMPONENTS], [fixed_width - x_coords[k] - widths[k] | k in COMPONENTS]));

% two items symmetry
constraint symmetry_breaking_constraint(
  forall(i,j in COMPONENTS where i<j)((x_coords[i]==x_coords[j] /\ actual_widths[i]==actual_widths[j]) -> y_coords[i]<=y_coords[j] - actual_heights[i]));
constraint symmetry_breaking_constraint(
  forall(i,j in COMPONENTS where i<j)((y_coords[i]==y_coords[j] /\ actual_heights[i]==actual_heights[j]) -> x_coords[i]<=x_coords[j] - actual_widths[i]));

% three items symmetry
constraint symmetry_breaking_constraint(
forall(i,j,k in COMPONENTS where i>j /\ j>k)((x_coords[i]==x_coords[j] /\ actual_widths[i]==actual_widths[j] /\ y_coords[i]==y_coords[k] /\ actual_heights[i] + actual_heights[j]==actual_heights[k]) -> x_coords[k]<=x_coords[i]));
constraint symmetry_breaking_constraint(
forall(i,j,k in COMPONENTS where i>j /\ j>k)((y_coords[i]==y_coords[j] /\ actual_heights[i]==actual_heights[j] /\ x_coords[i]==x_coords[k] /\ actual_widths[i] + actual_widths[j]==actual_widths[k]) -> y_coords[k]<=y_coords[i]));

% items of the same dimensions
constraint symmetry_breaking_constraint(
forall(i,j in COMPONENTS where j>i)(if widths [i]== widths [j] /\ heights [i]==heights [j] then x_coords[i] <= x_coords[j] /\ y_coords[i] <= y_coords[j] endif));
%-----------------------------------



solve :: seq_search([
int_search(y_coords, first_fail, indomain_min),
int_search(x_coords, first_fail, indomain_min),
int_search([height], smallest, indomain_min),
int_search(rot, smallest, indomain_min, complete)])
minimize height;

output[
    "x = ", show(x_coords), "\n",
    "y = ", show(y_coords), "\n",
    "rot = ", show(rot), "\n",
    "height = ", show(max(j in COMPONENTS)(y_coords[j] + actual_heights[j])), "\n"];
