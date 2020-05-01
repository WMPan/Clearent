module Graph
open MathNet.Numerics.LinearAlgebra
type Activiation = 
    |Act of string

type WM =
    |WMDouble of Matrix<double>
    |WMFloat of Matrix<float>
type VC =
    |VCDouble of Vector<double>
    |VCFloat of Vector<float>
type Layer = 
    |Singleton of Activiation*WM*VC
    |Gate of Activiation*WM*WM*VC
    |Cells of Layer list            //并行结构
    |Channel of Layer list          //顺序结构
    |Projection of int
    |Tensor of VC
    |Function of (VC list -> VC)
let (.*.) (a:Vector<double>) (b:Vector<double>) =  
    a.PointwiseMultiply(b)
let (.*) (a:Vector<double>) (b:Matrix<double>) = 
    b.Transpose().Multiply(a)
let Tanh (a:Vector<double>) = a.PointwiseTanh()

let LSTM1Model(wXI, wHI, BI, wXF, wHF, BF, 
                wXO, wHO, BO, wXC, wHC, BC, wHQ, BQ) =
    let s = Act("Sigmoid")
    let th = Act("Tanh")
    let I = Gate (s,WMDouble(wXI),WMDouble(wHI), VCDouble(BI))
    let F = Gate (s,WMDouble(wXF),WMDouble(wHF), VCDouble(BF))
    let O = Gate (s,WMDouble(wXO),WMDouble(wHO), VCDouble(BO))
    let Ct = Gate (th,WMDouble(wXC),WMDouble(wHC), VCDouble(BC))
    let funC fin = 
        match fin with
        |[VCDouble(A);VCDouble(B);VCDouble(D);VCDouble(cin)] 
            -> VCDouble( A .*. cin + B .*. D)
        |_-> failwith "haha"
    let funH fin = 
        match fin with
        |[VCDouble(B);VCDouble(A)] ->VCDouble(A .*. Tanh(B))
        |_-> failwith "haha"
    let funY inH = 
        match inH with
        |[VCDouble(H)] -> VCDouble(H .* wHQ + BQ)
        |_-> failwith "haha"

       
    let Layer1 = Cells [Projection(1); Projection(2)]
    let Layer2 = Cells [I; F; O; Ct]
    let Layer1_2 = Channel [Layer1; Layer2]
    let Layer11 = Projection(3)
    let Layer1_2_3 = Cells [Layer1_2; Layer11]
    let Layer3_1 = Cells [ Channel [Projection(1); Projection(2)];
                           Channel [Projection(1); Projection(1)]; 
                           Channel [Projection(1); Projection(4)];
                           Projection(2)]
    let Layer3_1_1 =Channel [Layer3_1; Function(funC)]
    let Layer3_2 = Channel [Projection(1); Projection(3)]
    let Layer4 = Cells [Layer3_1_1;Layer3_2]
    let Layer5 = Function(funH)
    let Layer6 = Function(funY)
    Channel [Layer1_2_3;Layer4;Layer5;Layer6]