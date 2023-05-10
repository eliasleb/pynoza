(* ::Package:: *)

(* ::Title::Initialization:: *)
(*Funknoza*)


(* ::Input::Initialization:: *)
Clear["Global`*"]
$PlotTheme="Scientific";


(* ::Input::Initialization:: *)
PositivePart[x_]:=If[x>=0,x,0];
c=1;
\[Mu]=1;
R = {x, y, z};
$Assumptions = t\[Element]Reals&x \[Element] Reals && y \[Element] Reals && z \[Element] Reals && \[Rho] \[Element] Reals && \[Rho] > 0 && \[Theta] \[Element] Reals && \[Phi] \[Element] Reals && k \[Element] Reals && k > 0 && \[Omega] > 0&&c\[Element]Reals&&c>0;
r = Sqrt[R . R];
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
\[Epsilon]=1/\[Mu]/c^2;
g[\[Alpha]_,f_,direction_] :=If[\[Alpha] == {0, 0, 0},
   1/4/\[Pi]/r f,
Block[{j, \[Alpha]1, ga},
\[Alpha]1 = \[Alpha];
j = firstNonzeroDim[\[Alpha]];
\[Alpha]1[[j]] = \[Alpha]1[[j]] - 1;
ga = g[\[Alpha]1,f,direction];
D[ga, R[[j]]] -direction R[[j]]/r D[ga,t]/c // Expand
]
];
h[\[Alpha]_,f_,direction_] :=g[\[Alpha],f,direction];
SimplifiedHeaviside[x_] := If[x >= 0, 1, 0];
SafeMoment[CurrentMoment_,\[Alpha]_,dim_]:=If[AllTrue[\[Alpha],#>=0&],
CurrentMoment[\[Alpha],dim],
0];
ChargeMoment[\[Alpha]_, CurrentMoment_, dim_,direction_] :=-direction  Sum[If[dim == j,
    \[Alpha][[j]] SimplifiedHeaviside[\[Alpha][[j]] - 2] (\[Alpha][[j]] - 1) Integrate[SafeMoment[CurrentMoment,(\[Alpha][[j]] - 2) UnitVector[3, j] +Total[ \[Alpha][[#]] UnitVector[3, #] & /@ Complement[{1, 2, 3}, {j }]], j],t],
    \[Alpha][[j]] \[Alpha][[dim]] SimplifiedHeaviside[\[Alpha][[j]] - 1] SimplifiedHeaviside[\[Alpha][[dim]] - 1] Integrate[SafeMoment[CurrentMoment,(\[Alpha][[j]] - 1) UnitVector[3, j] + (\[Alpha][[dim]] - 1) UnitVector[3, dim] + Total[\[Alpha][[#]] UnitVector[3, #] & /@ Complement[{1, 2, 3}, {dim, j}]], j],t]
    ],
   {j, 1, 3}]
ElectricMoment[\[Alpha]_, CurrentMoment_, dim_,direction_] := direction \[Mu] D[CurrentMoment[\[Alpha], dim],t] + 1/\[Epsilon] ChargeMoment[\[Alpha], CurrentMoment, dim,direction]
MagneticMoment[\[Alpha]_,CurrentMoment_, dim_,direction_] := -\[Mu] direction Sum[LeviCivitaTensor[3][[dim]][[i]][[j]] SimplifiedHeaviside[\[Alpha][[i]] - 1] \[Alpha][[i]] CurrentMoment[(\[Alpha][[i]] - 1) UnitVector[3, i] +Total[ \[Alpha][[#]] UnitVector[3, #] & /@ Complement[{1, 2, 3}, {i}]], j],
   {i, 1, 3}, {j, 1, 3}]
Field[\[Alpha]_, CurrentMoment_, MomentFunction_, dim_,direction_] :=(-1)^Norm[\[Alpha], 1]/(Times @@ (Factorial/@ \[Alpha])) h[\[Alpha],MomentFunction[\[Alpha], CurrentMoment, dim,direction],direction]
ElectricField[\[Alpha]_, CurrentMoment_, dim_,direction_] := Field[\[Alpha], CurrentMoment, ElectricMoment, dim,direction];
MagneticField[\[Alpha]_, CurrentMoment_, dim_,direction_] := Field[\[Alpha], CurrentMoment, MagneticMoment, dim,direction];
On[Assert]
CurrentMoment[\[Alpha]_, dim_] := If[\[Alpha] == {0, 0, 0} && dim == 3, Derivative[1][p][t], 0]
EfieldThis = Simplify@TransformedField["Cartesian" -> "Spherical", 
Table[
Sum[ElectricField[{ax, ay, az}, CurrentMoment, dim,-1],
      {ax, 0, 2}, {ay, 0, 2 - ax}, {az, 0, 2 - ax - ay}],
     {dim, 1, 3}], R -> {\[Rho], \[Theta], \[Phi]}]E^(\[ImaginaryJ] k \[Rho])/.
\[Integral]p[t]\[DifferentialD]t->1/\[ImaginaryJ]/\[Omega] p/.
(p'')[t]->-\[Omega]^2 p/.
Derivative[1][p][t]->\[ImaginaryJ] \[Omega] p/.
p[t]->p/.
\[Omega]->c k;
BfieldThis = Simplify@TransformedField["Cartesian" -> "Spherical", Table[Sum[MagneticField[{ax, ay, az}, CurrentMoment, dim,-1],
      {ax, 0, 2}, {ay, 0, 2 - ax}, {az, 0, 2 - ax - ay}],
     {dim, 1, 3}], R -> {\[Rho], \[Theta], \[Phi]}]E^(\[ImaginaryJ] k \[Rho])/.
\[Integral]p[t]\[DifferentialD]t->1/\[ImaginaryJ]/\[Omega] p/.
(p^\[Prime]\[Prime])[t]->-\[Omega]^2 p/.
Derivative[1][p][t]->\[ImaginaryJ] \[Omega] p/.
p[t]->p/.
\[Omega]->c k;
P = {0, 0, p};
n = R/r;
HJackson = TransformedField["Cartesian" -> "Spherical",
    c k^2/4/\[Pi] n\[Cross]P Exp[\[ImaginaryJ] k r]/r (1 - 1/\[ImaginaryJ]/k/r),
    R -> {\[Rho], \[Theta], \[Phi]}] // Simplify;
EJackson = TransformedField["Cartesian" -> "Spherical",
    1/4/\[Pi]/\[Epsilon] Exp[\[ImaginaryJ] k r] (k^2/r (n\[Cross]P)\[Cross]n + (3 n (n . P) - P) (1/r^3 - \[ImaginaryJ] k/r^2)),
    R -> {\[Rho], \[Theta], \[Phi]}] // Simplify;
Assert[Simplify[BfieldThis - HJackson \[Mu]] == {0, 0, 0}]
Assert[Simplify[EfieldThis - EJackson] == {0, 0, 0}]
Clear[CurrentMoment, EfieldThis, BfieldThis, P, n, HJackson, EJackson]


(* ::Input::Initialization:: *)
a=4;
length=2a;
(*    1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16*)
x1s={0,  0, 10,   0,    0, 45, 45, 68, 45,  45,    91,   91, 101,   91, 136, 136};
x2s={0, 10,20,  30,  40,  0, 20, 20, 30, 40,      0,    30,   20,   40,   10,     0};
ws={35, 10,23,  10,  35,10,23, 10, 10,  23,  10,   10,    23,   35,   10,   35};
hs={10, 10,10,  10,  10,20,10 ,30 ,10,  10,  20,   10,    10,    10,   40,    10};
(*           1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16*)
amplitudes={1,-4,-1,   4,   1,   1,   1,   1,    1,   1,       1,     1,     1,     1,      1,     1};
polarities={1,   2,  1,   2,   1,   2,   1,   2,    2,   1,       2,     2,     1,     1,      2,     1};
d1=a;
(*slice=1;;5;
x1s=x1s[[slice]];
x2s=x2s[[slice]];
ws=ws[[slice]];
hs=hs[[slice]];
amplitudes=amplitudes[[slice]];
polarities=polarities[[slice]];*)
Graphics[Flatten@
MapThread[
{{Red,Rectangle[{#1,#2},{#1+#3,#2+#4}]},
{Black,Text[{#5,#6,#7},{#1,#2}]}
}&
,{x1s,x2s,ws,hs,Range[Length@x1s],amplitudes,polarities}],
ImageSize->Small
]
lengthLogo=171;
d2=a*50/lengthLogo;
IntJ=Function[{x0,\[Sigma],m},
1/(m+1)((x0+\[Sigma]/2)^(m+1)-(x0-\[Sigma]/2)^(m+1))
];
CJ=Function[{\[Alpha],x1,x2,w,h},
IntJ[x1+w/2,w,\[Alpha][[1]]]IntJ[x2+h/2,h,\[Alpha][[2]]]
];
CurrentMoment=If[#1[[3]]==0,
Exp[-t^2] Sum[
If[polarities[[i]]==#2,
CJ[#1,x1s[[i]]/lengthLogo length-d1,x2s[[i]]/lengthLogo length-d2,ws[[i]]/lengthLogo length,hs[[i]]/lengthLogo length]amplitudes[[i]],
0],
{i,1,Length@x1s}],
0]&;
dirName=ToString@NotebookDirectory[];



(* ::Input::Initialization:: *)
ParallelTable[Block[{Efield,norm,plotLimit,nPoints,xd,yd,normd},
Efield=#1+#2&@@Table[
Table[
Sum[
ElectricField[{ax,ay,0},CurrentMoment,dim,direction]/.t->0-direction r/.z->0,
{ax,0,maxOrder},{ay,0,maxOrder-ax}],
{dim,1,2}],
{direction,-1,1,2}
];
norm=Total[Efield^2];
plotLimit=6/5a;
nPoints=15;
xd=Subdivide[-plotLimit,plotLimit,nPoints];
yd=Subdivide[-plotLimit,plotLimit,nPoints];
normd=norm/.x->xd/.y->yd;
Export[dirName<>"/data/order-"<>ToString[maxOrder]<>".mx",
<|"field"->Efield,"norm"->norm,"x"->xd,"y"->yd,"discrete norm"->normd|>];
],
{maxOrder,1,10,2}
]
