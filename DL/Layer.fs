module LanLayer
open ADiff
type InputShape =
    |InputShape1 of int
    |InputShape2 of int * int
    |InputShape3 of int * int*int

type Layer = 
    |Coda
    |DotActivation of Primitive
    |DotReLU
    |DotSigmoid
    |DotTanh 
    |Weight of int    //Weight List中的位置
    |DotPlus of Layer * Layer          //需要双输入
    |Dot of Layer * Layer              //需要双输入
    |Bias of int      //Bais List中的位置
    |Cells of Layer list            //并行结构
    |Channel of Layer list          //顺序结构
    |Projection of int              //向单个维度投影    
    |Variable of int                   //张量。。。。。    
    |Peep of int      //对循环指初值，指向List位置中的List   
    |Aside of int * Layer
    |Forward of int * Layer   //投影得到Recurrent张量输入
    |ErrorLayer of string

let Singleton a m b = //基本单元
    let IsSingleton =
        match a with
        |DotActivation s -> true
        |DotReLU -> true
        |DotSigmoid -> true
        |DotTanh -> true
        |_ -> false
    if IsSingleton then
        Channel [ DotPlus( Weight m, Bias b); a]
    else
        ErrorLayer "haha"

let Gate a w h b = //门单元
    let IsGate =
        match a with
        |DotActivation s -> true
        |DotReLU -> true
        |DotSigmoid -> true
        |DotTanh -> true
        |_ -> false
    if IsGate then
        let g0 = Channel [Projection 0; Weight w]
        let g1 = Channel [Projection 1; Weight h]
        Channel [DotPlus (DotPlus (g0, g1), Bias b); a]
    else
        ErrorLayer "haha"

let Scale w b = //比例变化，无激活函数
    DotPlus(Weight w, Bias b);

let LLSTM1 (i:int list) (a:int list) (w:int list) (b:int list) =   
    if i.Length <> 2 || a.Length <> 2  || w.Length <> 9 || b.Length <> 5 then
        failwith "haha"
    Channel [    
        Cells [Variable i.[0];     Peep a.[0];  Peep a.[1]];
        Channel [                
            Cells [ 
                Channel [
                    Cells [Projection 0; Projection 1];
                    Cells [
                        Gate DotSigmoid w.[0] w.[1] b.[0];
                        Gate DotSigmoid w.[2] w.[3] b.[1];
                        Gate DotSigmoid w.[4] w.[5] b.[2];
                        Gate DotTanh    w.[6] w.[7] b.[3]
                    ]
                ];
                Projection 2; //Cin
            ];   
            Cells [
                Projection 2; //O
                DotPlus(
                    Dot(Projection 1,Projection 4),
                    Dot(Projection 0,Projection 3)
                )
            ]
            Cells [
                Dot (Projection 0, Channel [Projection 1;DotTanh]);
                Projection 1
            ]
            Cells[
                Scale w.[8] b.[4];
                Projection 0;
                Projection 1
            ]

        ]; 
        Aside (a.[0],Projection 1);
        Aside (a.[1],Projection 2);
        Cells [Forward (i.[1],Projection 0)]       
    ]