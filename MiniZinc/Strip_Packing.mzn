include "globals.mzn";

% Loading the instance
int: n_components;   % Number of components
int: fixed_width; % Fixed width
set of int: COMPONENTS = 1..n_components;   % or anon_enum(n_components);
array[COMPONENTS] of int: widths; % components width(s)
array[COMPONENTS] of int: heights; % components height(s)

% Defining variables

array[COMPONENTS] of var 0..fixed_width-min(widths): x_coords; % lower left corner x coordinates
array[COMPONENTS] of var 0..2*max(min_h,max(heights))-min(heights): y_coords; % lower left corner y coordinates

int: min_h = (sum(k in COMPONENTS)(widths[k] * heights[k])) div fixed_width;

var min_h..2*max(min_h,max(heights)): height;

%-----------------------------------CONSTRAINTS
constraint diffn(x_coords, y_coords, widths, heights);
constraint cumulative(y_coords, heights, [widths[j] | j in COMPONENTS], fixed_width);
% constraint cumulative(x_coords, widths, [heights[j] | j in COMPONENTS], height);
constraint max(i in 1..n_components)(x_coords[i] + widths[i])<=fixed_width;
%constraint forall(i in 1..n_components)(x_coords[i] + widths[i]<=fixed_width);
constraint max(i in 1..n_components)(y_coords[i] + heights[i])<=height;
% constraint forall(i in 1..n_components)(y_coords[i] + heights[i]<=height);
%constraint symmetry_breaking_constraint(
%  lex_lesseq([x_coords[k] | k in COMPONENTS], [fixed_width - x_coords[k] - widths[k] | %k in COMPONENTS]));
%-----------------------------------

%-----------------------------------SYMMETRY BREAKING CONSTRAINTS
%constraint symmetry_breaking_constraint(lex_lesseq([x_coords[k] | k in COMPONENTS], [fixed_width - x_coords[k] - widths[k] | k in COMPONENTS]));


% two items symmetry
%constraint symmetry_breaking_constraint(
  %forall(i,j in COMPONENTS where i<j)((x_coords[i]==x_coords[j] /\ widths[i]==widths[j]) -> y_coords[i]<=y_coords[j] - heights[i]));
%constraint symmetry_breaking_constraint(
  %forall(i,j in COMPONENTS where i<j)((y_coords[i]==y_coords[j] /\ heights[i]==heights[j]) -> x_coords[i]<=x_coords[j] - widths[i]));


% three items symmetry
%constraint symmetry_breaking_constraint(
%forall(i,j,k in COMPONENTS where i>j /\ j>k)((x_coords[i]==x_coords[j] /\ widths[i]==widths[j] /\ y_coords[i]==y_coords[k] /\ heights[i] + heights[j]==heights[k]) -> x_coords[k]<=x_coords[i]));
%constraint symmetry_breaking_constraint(
%forall(i,j,k in COMPONENTS where i>j /\ j>k)((y_coords[i]==y_coords[j] /\ heights[i]==heights[j] /\ x_coords[i]==x_coords[k] /\ widths[i] + widths[j]==widths[k]) -> y_coords[k]<=y_coords[i]));


% items of identical dimensions
constraint symmetry_breaking_constraint(
forall(i,j in COMPONENTS where j>i)(if widths [i]== widths [j] /\ heights [i]==heights [j] then x_coords[i] <= x_coords[j] /\ y_coords[i] <= y_coords[j] endif));
%-----------------------------------

solve :: seq_search([
int_search(y_coords, first_fail, indomain_min),
int_search(x_coords, first_fail, indomain_min),
int_search([height], smallest, indomain_min)])
minimize height;



 output [
   if i=1 then show(fixed_width)++" "++show(height)++ "\n" endif ++
   if i=2 then show(n_components)++ "\n" endif ++
   if i>2 then show(widths[i-2])++" "++show(heights[i-2])++" "++show(x_coords[i-2])++" "++show(y_coords[i-2])++ "\n" endif
   | i in 1..n_components+2
];
