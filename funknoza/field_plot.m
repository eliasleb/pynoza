(* ::Package:: *)

(* ::Input::Initialization:: *)
(* Expect a=..., maxOrder=..., nPoints=... savePath *)
If[
Length@$ScriptCommandLine!=5,
Print["Expect command line arguments a=..., order=..., nPoints=... savePath"];Throw[_];
,_]
ToExpression@$ScriptCommandLine[[2]]
ToExpression@$ScriptCommandLine[[3]]
ToExpression@$ScriptCommandLine[[4]]
savePath=$ScriptCommandLine[[5]]


(* ::Input::Initialization:: *)
length = 2 a;

x1s={(*E*)0,0,10,0,0,(*P*)45,45,55,68,55,(*F*)91,91,101,91,(*L*)136,136};
x2s={(*E*)0,10,20,30,40,(*P*)0,20,20,30,40,(*F*)0,30,20,40,(*L*)10,0};
ws={(*E*)35,10,23,10,35,(*P*)10,10,23,10,23,(*F*)10,10,23,35,(*L*)10,35};
hs={(*E*)10,10,10,10,10,(*P*)20,30,10,10,10,(*F*)20,10,10,10,(*L*)40,10};
amplitudes={-2,8,6/2,-8,-2,(*P*)2,1,1,-1,2,(*F*)2,2,2,-2,(*L*)1,-1};
polarities={1,2,1,2,1,(*P*)2,2,1,2,1,(*F*)2,2,1,1,(*L*)2,1};

d1 = a;
d2 = a * 50 / lengthLogo;
lengthLogo = 171;

IntJ = Function[{x0, \[Sigma], m},
    1 / (m + 1) ((x0 + \[Sigma] / 2) ^ (m + 1) - (x0 - \[Sigma] / 2) ^ (m + 1))
];

CJ = Function[{\[Alpha], x1, x2, w, h},
    IntJ[x1 + w / 2, w, \[Alpha][[1]]] IntJ[x2 + h / 2, h, \[Alpha][[2]]]
];

CurrentMoment = If[#1[[3]] == 0,
    Exp[-t^2] Sum[
        If[polarities[[i]] == #2,
            CJ[#1, x1s[[i]] / lengthLogo length - d1, x2s[[i]] / lengthLogo
                 length - d2, ws[[i]] / lengthLogo length, hs[[i]] / lengthLogo length
                ] amplitudes[[i]]
            ,
            0
        ]
        ,
        {i, 1, Length @ x1s}
    ]
    ,
    0
]&;


(* ::Input::Initialization:: *)
Efield=0;
Table[
Efield+=#1+#2&@@ParallelTable[
Table[
Sum[
ElectricField[{ax,order-ax,0},CurrentMoment,dim,direction]/.t->0-direction r/.z->0,
{ax,0,order}],
{dim,1,2}
],{direction,-1,1,2}
];
Export[dirName<>"/data/field-order-"<>ToString[order]<>"-a-"<>ToString[a]<>".mx",
Efield],
{order,0,maxOrder}
]
