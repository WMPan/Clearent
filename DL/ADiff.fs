module ADiff
open System
type Primitive =
    |Zero
    |One
    |Number of double
    |V of int   
    |ADV of int * double 
    |Linear of double * Primitive
    |Power of Primitive * double
    |E of Primitive
    |Ln of Primitive
    |Sin of Primitive
    |Cos of Primitive
    |Tanh of Primitive
    |Sech of Primitive
    |Negative of Primitive
    |Plus of Primitive * Primitive
    |Subtract of Primitive * Primitive
    |Product of Primitive * Primitive
    |Divide of Primitive * Primitive



let rec ContainsV l i=
    match l with
    |Zero -> false
    |One -> false
    |Number( a) -> false
    |V( a) -> if a=i then true else false
    |ADV(a,b) -> if a=i then true else false
    |Linear(b,p) -> ContainsV p i
    |Power(p,b) -> ContainsV p i
    |E( p) -> ContainsV p i
    |Ln( p) -> ContainsV p i
    |Sin( p) -> ContainsV p i
    |Cos( p) -> ContainsV p i
    |Tanh( p) -> ContainsV p i
    |Sech( p) -> ContainsV p i
    |Negative( p) -> ContainsV p i
    |Plus(p,q) -> ContainsV p i || ContainsV q i
    |Subtract(p,q) -> ContainsV p i || ContainsV q i
    |Product(p,q) -> ContainsV p i || ContainsV q i
    |Divide(p,q) -> ContainsV p i || ContainsV q i



let rec Derivative l i=
    if ContainsV l i then
        match l with
        |Zero -> Zero
        |One -> Zero
        |Number( a) -> Zero
        |V( a) -> if a=i then One else Zero
        |ADV(a,b) -> if a=i then One else Zero
        |Linear(b,p) -> Linear(b,Derivative p i)
        |Power(p,b) -> Linear(b,Power(p,b - 1.))
        |E( p) -> Product(E( p),Derivative p i)
        |Ln( p) -> Divide(Derivative p i,p)
        |Sin( p) -> Product(Cos p,Derivative p i)
        |Cos( p) -> Product(Negative( Sin p),Derivative p i)
        |Tanh( p) -> Product(Product(Sech p,Sech p),Derivative p i)
        |Sech( p) -> Product(Negative( Product(Sech p,Tanh p)),Derivative p i)
        |Negative( p) -> Negative(Derivative p i)
        |Plus(p,q) -> Plus(Derivative p i, Derivative q i)
        |Subtract(p,q) -> Subtract(Derivative p i, Derivative q i)
        |Product(p,q) -> Plus(Product(Derivative p i, q), 
                                Product(p,Derivative q i))
        |Divide(p,q) -> Divide(Subtract(Product(Derivative p i, q), 
                                Product(p,Derivative q i)), Product(q,q))
    else
        Zero


let rec ChangeV l i=
    match l with
    |Zero -> Zero
    |One -> One
    |Number( a) -> Number( a)
    |V(a) -> V(i)
    |ADV(a,b) -> ADV(i,b)
    |Linear(b,p) -> Linear(b,ChangeV p i)
    |Power(p,b) -> Power(ChangeV p i,b)
    |E( p) -> E(ChangeV p i)
    |Ln( p) -> Ln(ChangeV p i)
    |Sin( p) -> Sin(ChangeV p i)
    |Cos( p) -> Cos(ChangeV p i)
    |Tanh( p) -> Tanh(ChangeV p i)
    |Sech( p) -> Sech(ChangeV p i)
    |Negative( p) -> Negative(ChangeV p i)
    |Plus(p,q) -> Plus(ChangeV p i,ChangeV q i)
    |Subtract(p,q) -> Subtract(ChangeV p i,ChangeV q i)
    |Product(p,q) -> Product(ChangeV p i,ChangeV q i)
    |Divide(p,q) -> Divide(ChangeV p i,ChangeV q i)



let rec Simplify f =  
    let rec ISimplify g =    
        let INumber b =
            match b with
            |0. -> Zero,true
            |1. -> One,true
            |_ -> Number(b), false
        let ILinear b p =
            match b,p with
            |(0.,p) -> Zero,true
            |(1.,p) -> Simplify p,true
            |(b,Zero) -> Zero,true
            |(b,One) -> Number(b),true
            |(b,Number(c)) -> Number(b * c),true
            |(b0,Linear(b1,p)) -> 
                Linear(b0 * b1,Simplify p),true     
            |_ -> 
                let p1,o = ISimplify p
                Linear(b,p1),o
        let IPower p b =
            match p,b with
            |(p,0.) -> One,true
            |(p,1.) -> p,true
            |(One,b) -> One,true
            |_ -> 
                let p1,o = ISimplify p
                Power(p1,b),o
        let IE b = 
            match b with
            |Zero -> One,true
            |_ -> E(b),false
        let ILn p =
            match p with
            |One -> Zero,true
            |_ -> 
                let p1,o = ISimplify p
                Ln(p1),o
        let INegative p =
            match p with
            |Zero -> Zero,true
            |_ -> 
                let p1,o = ISimplify p
                Negative(p1),o
        let IPlus p q =
            match p,q with
            |p,Zero -> p,true
            |Zero,q -> q,true
            |_ -> 
                let p1,op = ISimplify p
                let q1,oq = ISimplify q
                Plus(p1,q1),op||oq
        let ISubtract p q =
            match p,q with
            |p,Zero -> Simplify p,true
            |Zero,q -> Simplify (Negative q),true
            |_ ->
                let p1,op = ISimplify p
                let q1,oq = ISimplify q
                Subtract(p1,q1),op||oq
        let IProduct p q =
            match p,q with
            |Zero,q -> Zero,true
            |p,Zero -> Zero,true 
            |One,q -> Simplify q,true
            |p,One -> Simplify p,true 
            |_ ->
                let p1,op = ISimplify p
                let q1,oq = ISimplify q
                Product(p1,q1),op||oq
        let IDivide p q =
            match p,q with
            |Zero,q -> Zero,true
            |_ -> 
                let p1,op = ISimplify p
                let q1,oq = ISimplify q
                Divide(p1,q1),op||oq 
        let Inner l =
            match l with
            |Number(b) -> INumber b
            |Linear(b,p) -> ILinear b p
            |Power(p,b) -> IPower p b
            |E(b) -> IE b
            |Ln(p) -> ILn p
            |Sin(p) -> 
                let p1,o = ISimplify p
                Sin(p1),o
            |Cos(p) -> 
                let p1,o = ISimplify p
                Cos(p1),o
            |Tanh(p) -> 
                let p1,o = ISimplify p
                Tanh(p1),o
            |Sech(p) -> 
                let p1,o = ISimplify p
                Sech(p1),o
            |Negative(p) -> INegative p
            |Plus(p,q) -> IPlus p q
            |Subtract(p,q) -> ISubtract p q
            |Product(p,q) -> IProduct p q
            |Divide(p,q) -> IDivide p q       
            |_ -> l,false
        let g1,changed = Inner g
        let g2 = if changed = true then Simplify g1
                    else g1
        g2,changed
    let f1,changed = ISimplify f
    if changed = true then Simplify f1
    else f1

let rec Evaluate l (I:double list) :double =
    match l with
    |Zero -> 0.
    |One -> 1.
    |Number( a) -> a
    |V( a) -> I.[a]
    |ADV(a,b) -> b
    |Linear(b, p) -> b * (Evaluate p I)
    |Power(p,b) -> Math.Pow((Evaluate p I),b)
    |E( p) -> Math.Exp(Evaluate p I)
    |Ln( p) -> Math.Log(Evaluate p I)
    |Sin( p) -> Math.Sin(Evaluate p I)
    |Cos( p) -> Math.Cos(Evaluate p I)
    |Tanh( p) -> Math.Tanh(Evaluate p I)
    |Sech( p) -> 1./Math.Cosh(Evaluate p I)
    |Negative( p) -> -(Evaluate p I)
    |Plus(p,q) -> (Evaluate p I) + (Evaluate q I) 
    |Subtract(p,q) -> (Evaluate p I) - (Evaluate q I) 
    |Product(p,q) -> (Evaluate p I) * (Evaluate q I) 
    |Divide(p,q) -> (Evaluate p I) / (Evaluate q I) 

let rec Compile1VPrim l :double -> double =
    match l with
    |Zero -> fun v -> 0.
    |One -> fun v -> 1.
    |Number( a) -> fun v -> a
    |V( a) -> fun v -> v
    |ADV(a,b) -> fun v -> b
    |Linear(b, p) -> fun a -> b * (Compile1VPrim p a)
    |Power(p,b) -> fun a -> Math.Pow(Compile1VPrim p a, b)
    |E( p) -> fun a -> Math.Exp(Compile1VPrim p a)
    |Ln( p) -> fun a -> Math.Log(Compile1VPrim p a)
    |Sin( p) -> fun a -> Math.Sin(Compile1VPrim p a)
    |Cos( p) -> fun a -> Math.Cos(Compile1VPrim p a)
    |Tanh( p) -> fun a -> Math.Tanh(Compile1VPrim p a)
    |Sech( p) -> fun a -> 1./Math.Cosh (Compile1VPrim p a)    
    |Negative( p) -> fun a -> - (Compile1VPrim p a)
    |Plus(p,q) -> fun a -> (Compile1VPrim p a) + (Compile1VPrim q a)
    |Subtract(p,q) -> fun a -> (Compile1VPrim p a) - (Compile1VPrim q a)
    |Product(p,q) -> fun a -> (Compile1VPrim p a) * (Compile1VPrim q a) 
    |Divide(p,q) -> fun a -> (Compile1VPrim p a) / (Compile1VPrim q a) 