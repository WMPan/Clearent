module lstm 
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open FSharp.Data


let Sigmoid (a:Vector<double>) = 
    Vector<double>.Build.Dense
        ( a.Count ,
            (fun i -> 
                double (SpecialFunctions.Logistic (float a.[i]))))

let (.*) (a:Vector<double>) (b:Matrix<double>) = b.Transpose().Multiply(a)

let Tanh (a:Vector<double>) = a.PointwiseTanh()

let (.*.) (a:Vector<double>) (b:Vector<double>) =  a.PointwiseMultiply(b)




type LSTMParams(parameters:JsonValue, NumIn:int, NumHiddens:int, NumOut:int) =
    member this.WXI = 
        Matrix<double>.Build.Dense( NumIn, NumHiddens, 
            (fun i j -> (parameters.["WXI"].[i].[j].AsFloat())))
    member this.WHI = 
        Matrix<double>.Build.Dense( NumHiddens,NumHiddens, 
            (fun i j -> (parameters.["WHI"].[i].[j].AsFloat())))
    member this.BI = 
        Vector<double>.Build.Dense( NumHiddens, 
            (fun i -> (parameters.["BI"].[i].AsFloat())))
    member this.WXF = 
        Matrix<double>.Build.Dense( NumIn, NumHiddens, 
            (fun i j -> (parameters.["WXF"].[i].[j].AsFloat())))
    member this.WHF = 
        Matrix<double>.Build.Dense( NumHiddens,NumHiddens, 
            (fun i j -> (parameters.["WHF"].[i].[j].AsFloat())))
    member this.BF = 
        Vector<double>.Build.Dense( NumHiddens, 
            (fun i -> (parameters.["BF"].[i].AsFloat())))
    member this.WXO = 
        Matrix<double>.Build.Dense( NumIn, NumHiddens, 
            (fun i j -> (parameters.["WXO"].[i].[j].AsFloat())))
    member this.WHO = 
        Matrix<double>.Build.Dense( NumHiddens,NumHiddens,
            (fun i j -> (parameters.["WHO"].[i].[j].AsFloat())))
    member this.BO = 
        Vector<double>.Build.Dense( NumHiddens, 
            (fun i -> (parameters.["BO"].[i].AsFloat())))
    member this.WXC = 
        Matrix<double>.Build.Dense( NumIn, NumHiddens, 
            (fun i j -> (parameters.["WXC"].[i].[j].AsFloat())))
    member this.WHC = 
        Matrix<double>.Build.Dense( NumHiddens,NumHiddens, 
            (fun i j -> (parameters.["WHC"].[i].[j].AsFloat())))
    member this.BC = 
        Vector<double>.Build.Dense( NumHiddens, 
            (fun i -> (parameters.["BC"].[i].AsFloat())))
    member this.WHQ = 
        Matrix<double>.Build.Dense( NumHiddens,NumOut,
            (fun i j -> (parameters.["WHQ"].[i].[j].AsFloat())))
    member this.BQ = 
        Vector<double>.Build.Dense( NumOut, 
            (fun i -> (parameters.["BQ"].[i].AsFloat())))




let LSTM0(inputs: Vector<double> list, state, theparams) =
    match theparams with
        |(wXI, wHI, BI, wXF, wHF, BF, wXO, wHO, BO, wXC, wHC, BC, wHQ, BQ) ->

            let ( Hin, Cin) = state
            let mutable H = Hin
            let mutable C = Cin
            let mutable outputs = []
            for X in inputs do
                let I = Sigmoid (X .* wXI + Hin .* wHI + BI)
                let F = Sigmoid (X .* wXF + Hin .* wHF + BF)
                let O = Sigmoid (X .* wXO + Hin .* wHO + BO)
                let Ct = Tanh(X .* wXC  + H .* wHC + BC)
                C <- F .*. Cin + I .*. Ct
                H <- O .*. Tanh(C)
                let Y = H .* wHQ + BQ
                outputs <- outputs@[Y]
            (outputs, (H, C))
        |_ -> failwith "insufficient parama"

let OneHot (I:int ) (size:int ) =
    let X = Vector<double>.Build.Dense(size)
    X.[I] <- 1.0
    X

