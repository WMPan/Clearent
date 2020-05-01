module Layer
open TensorV
type InputShape =
    |InputShape1 of int
    |InputShape2 of int * int
    |InputShape3 of int * int*int
type ActivationType =
    |ReLU
    |SoftMax

type Kernel_Initializer_Type =
    |Glorot_uniform

type Bias_Initializer_Type =
    |Zeros

type ParameterDefinition =
    |Activation of ActivationType
    |Kernel_Initializer of Kernel_Initializer_Type
    |Bias_Initializer of Bias_Initializer_Type


type LayerDefinition =
    |Flaten of InputShape
    |Dense of int* ParameterDefinition list

type LayerParameter =
    |Input of Tensor
    |Output of Tensor
    |Kernal of Tensor
    |Bias of Tensor

type LayerData =
    |Input of Tensor
    |Output of Tensor
    |Memory of Tensor

type NNLayer =
    |Layer of LayerDefinition * LayerParameter list * LayerData list
    |Layers of NNLayer * NNLayer 


let GF (ld:LayerDefinition) (pl:LayerParameter list) (dl:LayerData list) = 0


let rec Compile (model:NNLayer) f shape = 
    let GF m sh = m,sh
    match model with
    |Layers(layers1,layers2) -> 
        let function1,shape1 = Compile layers1 f shape
        Compile layers2 (f>>function1) shape1
    |Layer(ld,pl,lds) -> 
        let shape2,function2 = GF model shape
        (f>>function2),shape

let a = Activation(ReLU)