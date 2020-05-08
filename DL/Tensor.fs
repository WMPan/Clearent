module TensorV
open MathNet.Numerics.LinearAlgebra



type PrecisionType =
    |DoublePrecision
    |FloatPrecision

type MyGenericClass<'a> (x: 'a) =
   do printfn "%A" x



let LayoutAt (shape:int list) (sp:int list) =
    let n = shape.Length
    if n <> sp.Length then failwith "haha"
    else
        let mutable m = 0
        let mutable c = 1
        for i in 0..(n - 1) do
            m <- m + sp.[i] * c
            c <- shape.[i] * c
        m

let Embed2D (v:Vector<double>) (a:double [,]) = 
    let i0 = a.GetLowerBound(0)
    let i0n = a.GetUpperBound(0)
    let j0 = a.GetLowerBound(1)
    let j0n = a.GetUpperBound(1)
    let s = [i0n - i0 + 1;
             j0n - j0 + 1]
    for i = i0 to i0n do
        for j = j0 to j0n do
            let n = LayoutAt s [i;j]
            v.[n] <- a.[i,j]

let Recover2D (v:Vector<double>)  (shape:int list) = 
    match shape with
    |[d0; d1] -> 
        Array2D.init d0 d1 (fun i j -> v.[LayoutAt shape [i;j]])
    |_ -> failwith "haha"



let Match2 (sm:(int*int) list) n = 
    let mutable i = [| for j in 0..(n-1) -> 0|]
    let mutable i0 = [| for j in 0..(n-1) -> 0|]
    let mutable a1 = 0
    let mutable a2 = 0
    for j in 0..sm.Length do
        i.[fst sm.[j]] <- snd sm.[j]
        i0.[fst sm.[j]] <- -1
    let mutable Break = false
    let mutable j1 = 0
    while not Break do
        if i0.[j1] <> -1 then
            a1 <- j1
            i0.[j1] <- -1
            Break <- true
        else j1 <- j1 + 1
    done       
    Break <- false
    let mutable j2 = 0
    while not Break do
        if i0.[j2] <> -1 then
            a2 <- j2
            Break <- true
        else j2 <- j1 + 1
    done 
    i, a1, a2

[<AllowNullLiteral>]
type Tensor(t) =
    let mutable N = 0
    let mutable t:PrecisionType = t
    let mutable s:int list = []
    let mutable ft:Vector<float> = null
    let mutable dt:Vector<double> = null
    new() = Tensor(DoublePrecision)
    member this.Ndim 
        with get() = N
        and set(value) = N <- value
    member this.Precision 
        with get() = t 
        and set(value) = t <- value
    member this.Shape 
        with get() = s
        and set(value) = s <- value
    member this.DT 
        with get() = dt 
        and set(value) = dt <- value
    member this.AsFloatVector() = 
        if N=1 && t = FloatPrecision then ft
        else failwith "haha"
    member this.AsDoubleVector() = 
        if N=1 && t = DoublePrecision then dt
        else failwith "haha"
    member this.FloatAt(sp) = 
        if N=2 then
            if t = FloatPrecision then ft
            else failwith "haha"
        else failwith "haha"
    member this.DoubleAt() = 
        if N=2 then
            if t = FloatPrecision then ft
            else failwith "haha"
        else failwith "haha"

    member this.DoubleMetrix() = 
        if N=2 then
            if t = DoublePrecision then ft
            else failwith "haha"
        else failwith "haha"
    member this.AsTensor(n1:int,n2:int,n3:int) = 
        N <- n1
        dt <- this.DoubleMetrix()
        this
