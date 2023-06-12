#!/usr/bin/env wolframscript
(* ::Package:: *)

Clear["Global`*"]
$PlotTheme="Scientific";
PositivePart[x_]:=If[x>=0,x,0];
R = {x, y, z};
$Assumptions = x \[Element] Reals && y \[Element] Reals && z \[Element] Reals && \[Rho] \[Element] Reals && \[Rho] > 0 && \[Theta] \[Element] Reals && \[Phi] \[Element] Reals && k \[Element] Reals && k > 0 && \[Omega] > 0;
r = Sqrt[R . R];
k=\[Omega];
predecessor[\[Alpha]_] := If[\[Alpha] == {0, 0, 0},
   {0, 0, 0},
   Select[
     Table[\[Alpha] - UnitVector[3, j], {j, 1, 3}],
     NonNegative@Min@# &,
     1
     ][[1]]
   ];
firstNonzeroDim[\[Alpha]_] := Position[
     \[Alpha],
     _?(# > 0 &),
     Infinity, 1
     ][[1]][[1]];
g[\[Alpha]_,direction_] :=g[\[Alpha],direction]=If[\[Alpha][[1]]<\[Alpha][[2]],
g[{\[Alpha][[2]],\[Alpha][[1]],\[Alpha][[3]]},direction]/.x->y1/.y->x/.y1->y,
 If[\[Alpha] == {0, 0, 0},
1/4/\[Pi]/r,
Block[{j, \[Alpha]1, ga},
\[Alpha]1 = \[Alpha];
j = firstNonzeroDim[\[Alpha]];
\[Alpha]1[[j]] = \[Alpha]1[[j]] - 1;
ga = g[\[Alpha]1,direction];
D[ga, R[[j]]] -direction \[ImaginaryJ] k R[[j]]/r ga //Expand
]
]
]
h[\[Alpha]_,direction_] :=g[\[Alpha],direction];
SimplifiedHeaviside[x_] := If[x >= 0, 1, 0];
SafeMoment[CurrentMoment_,\[Alpha]_,dim_]:=If[AllTrue[\[Alpha],#>=0&],
CurrentMoment[\[Alpha],dim],
0];
ChargeMoment[\[Alpha]_, CurrentMoment_, dim_,direction_] := -1/(direction \[ImaginaryJ] \[Omega]) Sum[If[dim == j,
    \[Alpha][[j]] SimplifiedHeaviside[\[Alpha][[j]] - 2] (\[Alpha][[j]] - 1) SafeMoment[CurrentMoment,(\[Alpha][[j]] - 2) UnitVector[3, j] +Total[ \[Alpha][[#]] UnitVector[3, #] & /@ Complement[{1, 2, 3}, {j }]], j],
    \[Alpha][[j]] \[Alpha][[dim]] SimplifiedHeaviside[\[Alpha][[j]] - 1] SimplifiedHeaviside[\[Alpha][[dim]] - 1] SafeMoment[CurrentMoment,(\[Alpha][[j]] - 1) UnitVector[3, j] + (\[Alpha][[dim]] - 1) UnitVector[3, dim] + Total[\[Alpha][[#]] UnitVector[3, #] & /@ Complement[{1, 2, 3}, {dim, j}]], j]
    ],
   {j, 1, 3}]
ElectricMoment[\[Alpha]_, CurrentMoment_, dim_,direction_] :=direction \[ImaginaryJ] \[Omega] CurrentMoment[\[Alpha], dim] +ChargeMoment[\[Alpha], CurrentMoment, dim,direction]
MagneticMoment[\[Alpha]_,CurrentMoment_, dim_,direction_] := -direction Sum[LeviCivitaTensor[3][[dim]][[i]][[j]] SimplifiedHeaviside[\[Alpha][[i]] - 1] \[Alpha][[i]] CurrentMoment[(\[Alpha][[i]] - 1) UnitVector[3, i] +Total[ \[Alpha][[#]] UnitVector[3, #] & /@ Complement[{1, 2, 3}, {i}]], j],
   {i, 1, 3}, {j, 1, 3}]
Field[\[Alpha]_, CurrentMoment_, MomentFunction_, dim_,direction_] := Block[{\[Beta]},
\[Beta]=(-1)^Norm[\[Alpha], 1]/(Times @@ (Factorial/@ \[Alpha])) MomentFunction[\[Alpha], CurrentMoment, dim,direction];
If[\[Beta]===0||\[Beta]===0.,
0,
\[Beta] h[\[Alpha],direction]
]
]
ElectricField[\[Alpha]_, CurrentMoment_, dim_,direction_] := Field[\[Alpha], CurrentMoment, ElectricMoment, dim,direction];
MagneticField[\[Alpha]_, CurrentMoment_, dim_,direction_] := Field[\[Alpha], CurrentMoment, MagneticMoment, dim,direction];
On[Assert]
CurrentMoment[\[Alpha]_, dim_] := If[\[Alpha] == {0, 0, 0} && dim == 3,
  \[ImaginaryJ] \[Omega] p, 0]
EfieldThis = Simplify@TransformedField["Cartesian" -> "Spherical", Table[Sum[ElectricField[{ax, ay, az}, CurrentMoment, dim,-1],
      {ax, 0, 2}, {ay, 0, 2 - ax}, {az, 0, 2 - ax - ay}],
     {dim, 1, 3}], R -> {\[Rho], \[Theta], \[Phi]}];
BfieldThis = Simplify@TransformedField["Cartesian" -> "Spherical", Table[Sum[MagneticField[{ax, ay, az}, CurrentMoment, dim,-1],
      {ax, 0, 2}, {ay, 0, 2 - ax}, {az, 0, 2 - ax - ay}],
     {dim, 1, 3}], R -> {\[Rho], \[Theta], \[Phi]}];
P = {0, 0, p};
n = R/r;
HJackson = TransformedField["Cartesian" -> "Spherical",
    k^2/4/\[Pi] n\[Cross]P/r (1 - 1/\[ImaginaryJ]/k/r),
    R -> {\[Rho], \[Theta], \[Phi]}]// Simplify;
EJackson = TransformedField["Cartesian" -> "Spherical",
    1/4/\[Pi] (k^2/r (n\[Cross]P)\[Cross]n + (3 n (n . P) - P) (1/r^3 - \[ImaginaryJ] k/r^2)),
    R -> {\[Rho], \[Theta], \[Phi]}] // Simplify;
Assert[Simplify[BfieldThis - HJackson] == {0, 0, 0}]
Assert[Simplify[EfieldThis - EJackson] == {0, 0, 0}]
Clear[CurrentMoment, EfieldThis, BfieldThis, P, n, HJackson, EJackson]

maxOrder=100;
rz=Sqrt[x^2+y^2];
antiderivative=Integrate[Exp[-t^2],t];
derivatives=Association@Table[
{i-1,dir}->Expand[FunctionExpand[
D[antiderivative,{t,i}]
]/.t->-dir rz],
{i,0,maxOrder+3},{dir,-1,1,2}];
ParallelTable[CurrentMoment[{ax,ay,0},dim],
{ax,0,maxOrder},{ay,0,maxOrder-ax},{dim,1,2}];
Table[Print[ax,ay,direction];g[{ax,ay,0},direction];,
{ax,0,maxOrder},{ay,0,maxOrder-ax},{direction,-1,1,2}];
DumpSave[ToString@Directory[];<>"data/g-up-to-order-"<>ToString@maxOrder<>".mx",g];



