//Infer_Cycling_0_2.fsx
(*Refs:
    C:\Users\inter\OneDrive\Projects(Comp)\Dev_2018\Infer.NET_2018\infer-master\test\TestFSharp
    https://github.com/dotnet/infer
    https://github.com/dotnet/infer/tree/master/test/TestFSharp
    https://dotnet.github.io/infer/InferNet101.pdf
*)
// #I @"C:\Users\inter\OneDrive\Projects(Comp)\Dev_2018\Infer.NET_2018\infer-master\test\TestFSharp\bin\Debug\net461"
#I @"C:\Users\inter\OneDrive\_myWork\Research2020\Infernet_2020\Packages"
// #r "netstandard.dll"
#r "Microsoft.ML.Probabilistic.Compiler.dll"
#r "Microsoft.ML.Probabilistic.dll"
#r "Microsoft.ML.Probabilistic.FSharp.dll"
#r "MathNet.Numerics.dll"
#r "NumSharp.Core.dll"
//#r "FSharp.Core.dll"

open System
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.Algorithms

open MathNet.Numerics

open NumSharp

//-----------------------------------------
// Logistic Regression Example 1:

let evidence = Variable.Bernoulli(0.5).Named("evidence")  
//IfBlock block = Variable.If(evidence); 
//block.CloseBlock();  
//InferenceEngine engine = new InferenceEngine();  
//double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;  
//Console.WriteLine("The probability that a Gaussian(0,1) > 0.5 is {0}", Math.Exp(logEvidence));


let data=[|Vector.FromArray[|1.0; -3.0; 1.0|];Vector.FromArray[|2.0; -2.1; 1.0|];Vector.FromArray[|1.0; -1.3; 1.0|];Vector.FromArray[|2.0; 0.5; 1.0|];Vector.FromArray[|1.0; 1.2; 1.0 |];
           Vector.FromArray[|1.0; 3.3; 1.0|];Vector.FromArray[|1.0; 4.4; 1.0|];Vector.FromArray[|1.0; 5.5;1.0|] |]
let rows = Range(data.Length)
let x = Variable.Constant(data, rows).Named("x")

let w = (Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray[|0.0; 0.0; 0.0 |], PositiveDefiniteMatrix.Identity(3))).Named("w")
let y = Variable.Array<bool>(rows)
let y= Variable.ArrayInit  rows (fun k-> Variable.BernoulliFromLogOdds(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x.[rows], w), 1.0)))
let engine = InferenceEngine(VariationalMessagePassing())

y.ObservedValue <- [|true; false; true; false; false; true; false; true |]

let postW = engine.Infer<VectorGaussian>(w)
(* VectorGaussian(-0.3902 0.02034 0.2653, 0.2634   0.009538 -0.3017 )
                                       0.009538 0.01458  -0.02437
                                       -0.3017  -0.02437 0.4693
*)

postW.GetLogProb (Vector.FromArray[|-0.3902464213; 0.02034384816; 0.2653339088|])
//val it : float = 1.134985956
postW.GetLogProb (Vector.FromArray[|-0.3902464213|])
//val it : float = 0.05619390925
postW.GetLogProb (Vector.FromArray[||])
//val it : float = -0.4308487086

let postWMean = postW.GetMean()
let postWVar = postW.GetVariance()
printfn "post W Mean: %A" postWMean
//post W Mean: seq [-0.3902464213; 0.02034384816; 0.2653339088]
printfn "post W Var: %O" postWVar
(*
post W Var: 0.2634   0.009538 -0.3017
            0.009538 0.01458  -0.02437
           -0.3017  -0.02437   0.4693
*)

//sqrt postWVar.[0,0]

let yPosterior = engine.Infer<Bernoulli[]>(y);

//val yPosterior : Bernoulli [] =
//  [|Bernoulli(1); Bernoulli(0); Bernoulli(1); Bernoulli(0); Bernoulli(0);Bernoulli(1); Bernoulli(0); Bernoulli(1)|]



//------------------------------------------
// Logistic Regression Example 2: (WIP)

let x1=np.linspace(-10.0, 10.0, 10000)
let x2=np.linspace(0.0, 20.0, 10000)
let bias = np.ones(x1.size)
let X = np.vstack([|x1;x2;bias|])
X.shape
//val it : int [] = [|3; 10000|]
let mutable B =  np.array([|-10.; 2.; 1.|]) // Sigmoid params for X + intercept


//let x1=Generate.LinearSpaced(10000, -10.0, 10.0)
//let x2=Generate.LinearSpaced(10000, 0.0, 20.0)
//let bias=Array.create x1.Length 1.0
//x1.Length,x2.Length,bias.Length

let logistic (x:NDArray,b:NDArray)=
  let mutable L=(x.T).dot(&b)*(-1.0)
  1.0/(1+np.exp(&L))

let pnoisy = logistic(X, B)

let r1=np.random.binomial(1,pnoisy.[0].GetDouble()).GetDouble()


(*
def logistic(x, b, noise=None):
    L = x.T.dot(b)
    if noise is not None:
        L = L+noise
    return 1/(1+np.exp(-L))

x1 = np.linspace(-10., 10, 10000)
x2 = np.linspace(0., 20, 10000)
bias = np.ones(len(x1))
X = np.vstack([x1,x2,bias]) # Add intercept
B =  [-10., 2., 1.] # Sigmoid params for X + intercept

# Noisy mean
pnoisy = logistic(X, B, noise=np.random.normal(loc=0., scale=0., size=len(x1)))
# dichotomize pnoisy -- sample 0/1 with probability pnoisy
y = np.random.binomial(1., pnoisy)
yl=list(y)

np.shape(y)

#plt.plot(y)
#plt.show()

#plt.hist(pnoisy)
#plt.show()
*)

//---------------------------------------------------------------------------------------
(*
// Charting
//---------
#I @"C:\Users\inter\OneDrive\Projects(Comp)\Dev_2018\Infer.NET_2018\packages"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open MathNet.Numerics
//open MathNet.Numerics.Distributions  //name conflict
//open MathNet.Numerics.Statistics

#load "FSharp.Charting.fsx"
open FSharp.Charting

let getValues (histogram:Statistics.Histogram) =
    let bucketWidth = Math.Abs(histogram.LowerBound - histogram.UpperBound) / (float histogram.BucketCount)
    [0..(histogram.BucketCount-1)]
    |> Seq.map (fun i -> (histogram.Item(i).LowerBound + histogram.Item(i).UpperBound)/2.0, histogram.Item(i).Count)
//--------------------------------------------------------------------------------------
*)
//--------------------------------------------------------------------
//Extras
//------

//Gamma------------------------------------
//let dist1=Distributions.Gamma(shape=2.0,rate=1./0.5)
//let samples1 = dist1.Samples() |> Seq.take 10000 |> Seq.toList
//let histogram = Statistics.Histogram(samples1, 100)
//Chart.Column (getValues histogram)
