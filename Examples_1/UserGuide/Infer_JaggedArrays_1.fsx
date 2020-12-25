//Infer_Multinomial.fsx <WIP>
(*Refs:
    https://dotnet.github.io/infer/userguide/Jagged%20arrays.html
*)

#I @"C:\Users\inter\OneDrive\_myWork\Research2020\Infernet_2020\Packages"
#r "Microsoft.ML.Probabilistic.Compiler.dll"
#r "Microsoft.ML.Probabilistic.dll"
#r "Microsoft.ML.Probabilistic.FSharp.dll"
#r "MathNet.Numerics.dll"

open System
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.Algorithms


//------------------
//Test Jagged Arrays
let sizes = [|2; 3|]  
let item = Range(sizes.Length).Named("item")
let sizesVar = Variable.Constant(sizes, item).Named("sizes");  
let feature = Range(sizesVar.[item]).Named("feature");
let x = Variable.Array<double>(Variable.Array<double>(feature), item).Named("x")
let xPrior = Gaussian(1.2, 3.4)

let x1=Variable.Random(xPrior).ForEach(item, feature)
Variable.ConstrainPositive(x1)

//------------------
//Constant jagged arrays
let a=[| [| 1.0; 2.0 |];
        [| 3.0; 4.0 |] |]
//val a : float [] [] = [|[|1.0; 2.0|]; [|3.0; 4.0|]|]
let innerSizes = Array.zeroCreate<int> a.Length  

for i in 0..(a.Length-1) do  
    innerSizes.[i] <- a.[i].Length

let outer = Range(a.Length).Named("outer")
let innerSizesVar = Variable.Constant(innerSizes, outer).Named("innerSizes")  

let inner = Range(innerSizesVar.[outer]).Named("outer")  
let aConst = Variable.Constant(a, outer, inner)

//------------------
//Observed jagged arrays
let outerSizeVar = Variable.New<int>()
let outer2 = Range(outerSizeVar)
let innerSizesVar2 = Variable.Array<int>(outer2)
let inner2 = Range(innerSizesVar.[outer2])  
let aObs = Variable.Array<double>(Variable.Array<double>(inner2), outer2)

let a2=[| [| 1.1; 3.3 |];
        [| 1.1;2.2; 4.4 |] |]
outerSizeVar.ObservedValue <- a2.Length
let innerSizes2 = Array.zeroCreate<int> a2.Length
for i in 0..(a2.Length-1) do  
    innerSizes2.[i] <- a2.[i].Length
innerSizesVar.ObservedValue <- innerSizes2
aObs.ObservedValue <- a2

//------------------
//More complex jagged arrays
let sizes2D = array2D [ [ 2; 3]; [4; 2]; [3; 1] ]
let rx = Range(sizes2D.GetLength(0)).Named("rx")  
let ry = Range(sizes2D.GetLength(1)).Named("ry")  
let sizes2DVar = Variable.Constant(sizes2D, rx, ry)
let rz = Range(sizes2DVar.[rx,ry]).Named("rz")
let zVar = Variable.Array(Variable.Array<double>(rz), rx, ry).Named("zVar")
let a3 = Variable.Array<Vector>(Range(1))  
let b = Variable.Array<VariableArray<Vector>, Vector[][]>(a3, Range(2))  
let c = Variable.Array<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]>(b, Range(3))
let d = Variable.Array<VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]>, Vector[][][][]>(c, Range(4))

type VarVectArr2 = VariableArray<VariableArray<Vector>, Vector[][]>
//let bb = Variable.Array<VarVectArr2>(a3, Range(2))  //Fail
