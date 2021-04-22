module PreTrained
open MathNet.Numerics.LinearAlgebra
let num_hiddens = 256
let ini_H = Vector<double>.Build.Dense(num_hiddens) 
ini_H.[5] <- 1.
ini_H.[6] <- 8.
let ini_C = Vector<double>.Build.Dense(num_hiddens)
ini_C.[200] <- 1.
ini_C.[6] <- 8.

let ap =[("wXI",0);("wHI",1);("wXF",2);("wHF",3);
          ("wXO",4);("wHO",5);("wXC",6);("wHC",7);
          ("wHQ",8);
          ("BI",0);("BF",1);("BO",2);("BC",3);
          ("BQ",4);
          ("X",0);
          ("Y",1);
          ("H",0);
          ("C",1)
         ] 

let paras = Map.ofList ap

let (wXI, wHI, BI, wXF, wHF, BF, wXO, wHO, BO, wXC, wHC, BC, wHQ, BQ) 
    = lstm.theParams
CL.WB.[paras.["wXI"]] <- wXI
CL.WB.[paras.["wHI"]] <- wHI
CL.WB.[paras.["wXF"]] <- wXF
CL.WB.[paras.["wHF"]] <- wHF
CL.WB.[paras.["wXO"]] <- wXO
CL.WB.[paras.["wHO"]] <- wHO
CL.WB.[paras.["wXC"]] <- wXC
CL.WB.[paras.["wHC"]] <- wHC
CL.WB.[paras.["wHQ"]] <- wHQ

CL.BB.[paras.["BI"]] <- BI
CL.BB.[paras.["BF"]] <- BF
CL.BB.[paras.["BO"]] <- BO
CL.BB.[paras.["BC"]] <- BC
CL.BB.[paras.["BQ"]] <- BQ

let w =  [ paras.["wXI"];paras.["wHI"];
            paras.["wXF"];paras.["wHF"];
            paras.["wXO"];paras.["wHO"];
            paras.["wXC"];paras.["wHC"];
            paras.["wHQ"]]
let b =  [paras.["BI"];
            paras.["BF"];
            paras.["BO"];
            paras.["BC"];
            paras.["BQ"]]
let i =  [ paras.["X"];
            paras.["Y"]]
let a =  [ paras.["H"];
            paras.["C"]]

let model = LanLayer.LLSTM1 i a w b
CL.VB.[paras.["X"]] <- lstm.CharToOneHot "好"
CL.AB.[paras.["H"]] <- ini_H
CL.AB.[paras.["C"]] <- ini_C

let fmodel = CL.CompileLayer model

let LSTM1Run1 ai =
    let cv =(fmodel []).[0]
    let c = lstm.OneHotToChar cv
    CL.VB.[paras.["X"]] <- lstm.CharToOneHot c
    c
let LSTM1Run (ai:string) n =
    CL.AB.[paras.["H"]] <- ini_H
    CL.AB.[paras.["C"]] <- ini_C
    let mutable b = ""
    for c in ai do
        CL.VB.[paras.["X"]] <- lstm.CharToOneHot (c.ToString())
        b <- lstm.OneHotToChar (fmodel []).[0]


    CL.VB.[paras.["X"]] <- lstm.CharToOneHot b
    b <- ai + b
    for i in ai.Length..n do
        let cv = (fmodel []).[0]
        let c = lstm.OneHotToChar cv
        CL.VB.[paras.["X"]] <- lstm.CharToOneHot (c.ToString())
        b <- b + c
    b
