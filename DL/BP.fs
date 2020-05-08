module BP
open LanLayer
open ADiff
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics
open System

let NuDotSigmoid (a:Vector<double>) = 
        Vector<double>.Build.Dense
            ( a.Count ,
                (fun i -> 
                    double (SpecialFunctions.Logistic (a.[i]))))
let NuDotTanh (a:Vector<double>) = a.PointwiseTanh()
let NuDotDeSigmoid (a:Vector<double>) = 
        Vector<double>.Build.Dense
            ( a.Count ,
                (fun i -> 
                    (Math.Pow(Math.E,-a.[i])/
                        Math.Pow(1. + Math.Pow(Math.E,-a.[i]),2.)) ))
let NuDotDeTanh (a:Vector<double>) =  a.PointwiseCosh().PointwisePower(-2.)
let NuDotReLU (a:Vector<double>) = 
        Vector<double>.Build.Dense
            ( a.Count ,
                (fun i -> 
                    if a.[i] >= 0. then a.[i] else 0.))
let NuDotDeReLU (a:Vector<double>) = 
        Vector<double>.Build.Dense
            ( a.Count ,
                (fun i -> 
                    if a.[i] >= 0. then 1. else 0.))
let NuDotOfFunction (f:double -> double) (a:Vector<double>) = 
        Vector<double>.Build.Dense
            ( a.Count , (fun i -> f a.[i]))      
let NuMMultiply (a:Matrix<double>) (b:Vector<double>) = a.Multiply(b)
let NuDot (a:Vector<double>) (b:Vector<double>) =  a.PointwiseMultiply(b)
let NuAdd (a:Vector<double>) (b:Vector<double>) =  a.Add(b)


let (VB :Vector<double> array) = Array.init 1000 (fun index -> null)
let (AB :Vector<double> array) = Array.init 1000 (fun index -> null)
let (WB :Matrix<double> array) = Array.init 1000 (fun index -> null)
let (BB :Vector<double> array) = Array.init 1000 (fun index -> null)

type VlToVl = Vector<double> list -> Vector<double> list 

let CompileLayer l =
    let rec Listfy f = //n->n
        let rec IListfy  fl (vInL:Vector<double> list) =  
            let v = vInL.[0]
            let v1 = fl v
            [v1]
        IListfy f
    let rec Listfy2 fs =
        let ILF (imfs:VlToVl list) (vs:Vector<double> list) = 
            (imfs.[0] vs)@(imfs.[1] vs)             
        ILF fs 
    let rec Listfy2To1 f =
        let ILF imf (vs:Vector<double> list) = 
            [imf vs.[0] vs.[1]]             
        ILF f
    
    let rec ICompileLayer Il =
        let rec CFlat fs = // n->? @
            let rec IMap ifs ls =  
                match ifs with
                |hf::tif -> (ICompileLayer hf ls)@IMap tif ls
                |_ -> []
            IMap fs
        let rec CCompose lys =
            match lys with
            |h::tail -> (ICompileLayer h)>>(CCompose tail)
            |[] -> fun a -> a
        match Il with
        |DotActivation s -> Listfy (NuDotOfFunction (Compile1VPrim s))
        |DotReLU -> Listfy NuDotReLU
        |DotSigmoid -> Listfy NuDotSigmoid
        |DotTanh -> Listfy NuDotTanh
        |Weight n -> Listfy (NuMMultiply WB.[n])
        |DotPlus (p,q) -> 
            let B = Listfy2 [ICompileLayer p;ICompileLayer q]
            B>>(Listfy2To1 NuAdd)     
        |Dot (p,q) ->                
            let B = Listfy2 [ICompileLayer p;ICompileLayer q]
            B>>(Listfy2To1 NuDot) 
        |Bias n -> Listfy (fun a -> BB.[n])  
        |Variable n -> fun a -> [VB.[n]]   
        |Peep n -> fun a -> [AB.[n]]           
        |Aside(n,ly) -> 
            let f = ICompileLayer ly
            let Asidesf (vs:Vector<double> list) =
                let result = f vs
                AB.[n] <- result.[0]
                vs
            Asidesf 
        |Cells ls -> CFlat ls  
        |Channel lys -> CCompose lys        
        |Projection n -> fun ls -> [ls.[n]]              
        |Coda -> fun a -> a   
        |Forward (n,ly) -> 
            let f = ICompileLayer ly
            let Asidesf (vs:Vector<double> list) =
                let result = f vs
                VB.[n] <- result.[0]
                result
            Asidesf  
        |ErrorLayer s -> failwith "haha"
    ICompileLayer l