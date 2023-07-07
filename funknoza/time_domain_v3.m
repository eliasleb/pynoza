(* ::Package:: *)

(* ::Input::Initialization:: *)
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
g[\[Alpha]_,direction_] :=If[\[Alpha][[1]]<\[Alpha][[2]],
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

(* Expect a=..., order=..., savePath *)
On[Assert];
$AssertFunction = Abort[]&;
Assert[Length@$ScriptCommandLine==5]
ToExpression@$ScriptCommandLine[[2]]
ToExpression@$ScriptCommandLine[[3]]
savePath=$ScriptCommandLine[[4]]
Print[a,order,savePath]


(* ::Input::Initialization:: *)
x1s={(*E*)0,0,10,0,0,(*P*)45,45,55,68,55,(*F*)91,91,101,91,(*L*)136,136};
x2s={(*E*)0,10,20,30,40,(*P*)0,20,20,30,40,(*F*)0,30,20,40,(*L*)10,0};
ws={(*E*)35,10,23,10,35,(*P*)10,10,23,10,23,(*F*)10,10,23,35,(*L*)10,35};
hs={(*E*)10,10,10,10,10,(*P*)20,30,10,10,10,(*F*)20,10,10,10,(*L*)40,10};
amplitudes={-2,8,6/2,-8,-2,(*P*)8/5,4/5,4/5,-4/5,8/5,(*F*)14/5,14/5,14/5,-14/5,(*L*)7/4,-7/4};
polarities={1,2,1,2,1,(*P*)2,2,1,2,1,(*F*)2,2,1,1,(*L*)2,1};
d1 = a;
lengthLogo = 171;

d2 = a * 50 / lengthLogo;
IntJ=Function[{x0,\[Sigma],m},
1/(m+1)((x0+\[Sigma]/2)^(m+1)-(x0-\[Sigma]/2)^(m+1))
];
CJ=Function[{\[Alpha],x1,x2,w,h},
IntJ[x1+w/2,w,\[Alpha][[1]]]IntJ[x2+h/2,h,\[Alpha][[2]]]
];
Clear[CurrentMoment];
CurrentMoment[\[Alpha]_,dim_]:=CurrentMoment[\[Alpha],dim]=If[\[Alpha][[3]]==0,
I \[Omega] Sum[
If[polarities[[i]]==dim,
CJ[\[Alpha],x1s[[i]]/lengthLogo length-d1,x2s[[i]]/lengthLogo length-d2,ws[[i]]/lengthLogo length,hs[[i]]/lengthLogo length]amplitudes[[i]],
0],
{i,1,Length@x1s}],
0];
maxOrder=120;
rz=Sqrt[x^2+y^2];
antiderivative=Integrate[Exp[-t^2],t];
derivatives=Association@Table[
{i-1,dir}->Expand[FunctionExpand[
D[antiderivative,{t,i}]
]/.t->-dir rz],
{i,0,maxOrder+3},{dir,-1,1,2}];
ParallelTable[CurrentMoment[{ax,ay,0},dim],
{ax,0,maxOrder},{ay,0,maxOrder-ax},{dim,1,2}];


(* ::Input::Initialization:: *)
Efield=Import[dirName<>"data/field-v3-order-"<>ToString@order<>"-a-"<>ToString@a<>".mx"];
Efield+=#1+#2&@@ParallelTable[
Table[
Sum[
Total@MapIndexed[{expr,index}|->expr derivatives[{index[[1]]-2,direction}]/(I)^(index[[1]]-1),
CoefficientList[
ElectricField[{ax,order-ax,0},CurrentMoment,dim,direction]/.z->0,\[Omega]]],
{ax,0,order}],
{dim,1,2}
],{direction,-1,1,2}
];
Efield=Expand@Efield;
Export[
dirName<>"data/field-v3-order-"<>ToString[order]<>"-a-"<>ToString[a]<>".mx",
Efield
]
