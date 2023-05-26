(* ::Package:: *)

(* ::Input::Initialization:: *)
If[
Length@$ScriptCommandLine!=4,
Print["Expect command line arguments a=..., order=..., nPoints=..."];Throw[_];
,_]
ToExpression@$ScriptCommandLine[[2]]
ToExpression@$ScriptCommandLine[[3]]
ToExpression@$ScriptCommandLine[[4]]


(* ::Input::Initialization:: *)
Efield=Import[ToString@Directory[]<>"/data/field-order-"<>ToString[order]<>"-a-"<>ToString[a]<>".mx"];
norm=Total[Efield^2];
plotLimitX=6/5a;
plotLimitY=2a 50/171;
nPointsX=21;
nPointsY=Round[nPointsX plotLimitY/plotLimitX];
xd=Subdivide[-plotLimitX,plotLimitX,nPointsX];
yd=Subdivide[-plotLimitY,plotLimitY,nPointsY];
normd=ParallelTable[
norm/.x->xd[[i]]/.y->yd[[j]],
{i,1,Length@xd},{j,1,Length@yd}];
Export[ToString@Directory[]<>"/data/order-"<>ToString[order]<>"-a-"<>ToString[a]<>".mx",
<|"norm"->norm,"x"->xd,"y"->yd,"discrete norm"->normd|>];
