module lstm 
open System.Collections.Generic
open MathNet.Numerics.LinearAlgebra
open FSharp.Data
open BP

let a = JsonValue.Load( "/Users/panwuming/datasets/lyricsdatanew.json")
let p = JsonValue.Load( "/Users/panwuming/datasets/LSTMParamsnew.json")


type DataLyrics(corpus_indices, char_to_idx, idx_to_char, vocab_size) =    
        member this.CorpusIndices = corpus_indices
        member this.CharToIdx:Dictionary<string, int> = char_to_idx
        member this.IdxToChar:array<string> = idx_to_char
        member this.VocabSize = vocab_size
let I1 = ([| for v in a.["CorpusIndices"] -> (v.AsInteger() )|])
let dict2 = new Dictionary<string, int>()
for (e,f) in a.["CharToIdx"].Properties() do
    dict2.Add(e,f.AsInteger())
let arr = a.["IdxToChar"].AsArray()
let nr = arr.Length - 1
let ar3 = [|for i in 0..nr -> (arr.[i]).AsString() |]
let v4 = a.["VocabSize"].AsInteger()
let DLyrics = DataLyrics(I1, dict2, ar3,v4)

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

let LP = LSTMParams(p,DLyrics.VocabSize,256,DLyrics.VocabSize)

let OneHot (I:int ) (size:int ) =
    let X = Vector<double>.Build.Dense(size)
    X.[I] <- 1.0
    X
let CharToOneHot c =
    OneHot DLyrics.CharToIdx.[c] DLyrics.VocabSize
let OneHotToChar (o:Vector<double>) =
    let y:int = o.MaximumIndex()
    if y < DLyrics.VocabSize then DLyrics.IdxToChar.[y]
    else failwith "不存在的字符" 

let theParams = ( LP.WXI.Transpose(), LP.WHI.Transpose(), LP.BI,
                  LP.WXF.Transpose(), LP.WHF.Transpose(), LP.BF,
                  LP.WXO.Transpose(), LP.WHO.Transpose(), LP.BO, 
                  LP.WXC.Transpose(), LP.WHC.Transpose(), LP.BC,
                  LP.WHQ.Transpose(), LP.BQ)

let (wXI, wHI, BI, wXF, wHF, BF, wXO, wHO, BO, wXC, wHC, BC, wHQ, BQ) = theParams
let LSTM1(X: Vector<double>, state) =
    let ( Hin, Cin) = state
    let I = NuDotSigmoid ((NuMMultiply wXI X)  + (NuMMultiply wHI Hin)  + BI)
    let F = NuDotSigmoid ((NuMMultiply wXF X) + (NuMMultiply wHF Hin) + BF)
    let O = NuDotSigmoid ((NuMMultiply wXO X) + (NuMMultiply wHO Hin) + BO)
    let Ct = NuDotTanh((NuMMultiply wXC X) + (NuMMultiply wHC Hin) + BC)
    let C = NuDot F Cin + NuDot I  Ct
    let H = NuDot O (NuDotTanh(C))
    let Y = NuMMultiply wHQ H + BQ
    (Y, (H, C))


