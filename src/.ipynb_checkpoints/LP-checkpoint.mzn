% Use this editor as a MiniZinc scratch book
var 0..100: b;
var 0..100: c;

%flour
constraint 250*b + 200*c <= 4000;

%bananas
constraint 2*b <= 6;

%sugar
constraint 75*b + 150*c <= 2000;

%butter
constraint 100*b + 150*c <= 500;

%cocoa
constraint 75*c <= 500;

%maximize our profit 
solve maximize 400*b + 450*c;

output["np. of banana cakes = \(b)\n",
  "no. of chococolate cakes = \(c)\n"];