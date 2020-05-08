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
BP.WB.[paras.["wXI"]] <- wXI
BP.WB.[paras.["wHI"]] <- wHI
BP.WB.[paras.["wXF"]] <- wXF
BP.WB.[paras.["wHF"]] <- wHF
BP.WB.[paras.["wXO"]] <- wXO
BP.WB.[paras.["wHO"]] <- wHO
BP.WB.[paras.["wXC"]] <- wXC
BP.WB.[paras.["wHC"]] <- wHC
BP.WB.[paras.["wHQ"]] <- wHQ

BP.BB.[paras.["BI"]] <- BI
BP.BB.[paras.["BF"]] <- BF
BP.BB.[paras.["BO"]] <- BO
BP.BB.[paras.["BC"]] <- BC
BP.BB.[paras.["BQ"]] <- BQ

//BP.VB.[paras.["X"]] 
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
BP.VB.[paras.["X"]] <- lstm.CharToOneHot "好"
BP.AB.[paras.["H"]] <- ini_H
BP.AB.[paras.["C"]] <- ini_C

let fmodel = BP.CompileLayer model

let LSTM1Run1 ai =
    let cv =(fmodel []).[0]
    let c = lstm.OneHotToChar cv
    BP.VB.[paras.["X"]] <- lstm.CharToOneHot c
    c
let LSTM1Run (ai:string) n =
    BP.AB.[paras.["H"]] <- ini_H
    BP.AB.[paras.["C"]] <- ini_C
    let mutable b = ""
    for c in ai do
        BP.VB.[paras.["X"]] <- lstm.CharToOneHot (c.ToString())
        b <- lstm.OneHotToChar (fmodel []).[0]


    BP.VB.[paras.["X"]] <- lstm.CharToOneHot b
    b <- ai + b
    for i in ai.Length..n do
        let cv = (fmodel []).[0]
        let c = lstm.OneHotToChar cv
        BP.VB.[paras.["X"]] <- lstm.CharToOneHot (c.ToString())
        b <- b + c
    b
        