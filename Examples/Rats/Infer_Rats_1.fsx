//PLL_infer_4.fsx
//RATS (Hierarchical Model)
//=========================

(*Refs:
    https://github.com/dotnet/infer/blob/master/src/Tutorials/BugsRats.cs
    C:\Users\inter\OneDrive\Projects(Comp)\Dev_2018\Infer.NET_2018\infer-master\
*)
#I @"C:\Users\inter\OneDrive\Projects(Comp)\Dev_2018\Infer.NET_2018\infer-master\test\TestFSharp\bin\Debug\net461"

#r "netstandard.dll"

#r "Microsoft.ML.Probabilistic.Compiler.dll"
#r "Microsoft.ML.Probabilistic.dll"
#r "Microsoft.ML.Probabilistic.FSharp.dll"

#r "FSharp.Core.dll"


open System
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math

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


//-----------------------------------------------------------------------------------
// Infer.NET: F# script for Rats (BUGS)
//-----------------------------------------------------------------------------------

(*
   Y_ij ~ Normal( a_i + b_i*(x_j - xbar ), tc )

    a_i ~ Normal( ac , ta )

    b_i ~ Normal( bc , tb )

xbar = 22
a0 = ac - bc*xbar
ac , ta , bc , tb , tc are given independent ``noninformative'' priors.
*)

(*
30 rats whose weights were measured at each of five consecutive weeks.

Model
Weights are modeled as

y_{i,j} &\sim \text{Normal}\left(\alpha_i + \beta_i (x_j - \bar{x}), \sigma_c\right) \quad\quad i=1,\ldots,30; j=1,\ldots,5 \\
\alpha_i &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
\beta_i &\sim \text{Normal}(\mu_\beta, \sigma_\beta) \\
\mu_\alpha, \mu_\beta &\sim \text{Normal}(0, 1000) \\
\sigma^2_\alpha, \sigma^2_\beta, \sigma^2_c &\sim \text{InverseGamma}(0.001, 0.001),

where y_{i,j} is repeated weight measurement j on rat i, and x_j is the day on which the measurement was taken.

*)

//The Data--------------------------------------------
// Height data
let RatsHeightData:double[,] = array2D [
        [ 151.0; 199.0; 246.0; 283.0; 320.0 ];
       [ 145.0; 199.0; 249.0; 293.0; 354.0 ];
       [ 147.0; 214.0; 263.0; 312.0; 328.0 ];
       [ 155.0; 200.0; 237.0; 272.0; 297.0 ];
       [ 135.0; 188.0; 230.0; 280.0; 323.0 ];
       [ 159.0; 210.0; 252.0; 298.0; 331.0 ];
       [ 141.0; 189.0; 231.0; 275.0; 305.0 ];
       [ 159.0; 201.0; 248.0; 297.0; 338.0 ];
       [ 177.0; 236.0; 285.0; 350.0; 376.0 ];
       [ 134.0; 182.0; 220.0; 260.0; 296.0 ];
       [ 160.0; 208.0; 261.0; 313.0; 352.0 ];
       [ 143.0; 188.0; 220.0; 273.0; 314.0 ];
       [ 154.0; 200.0; 244.0; 289.0; 325.0 ];
       [ 171.0; 221.0; 270.0; 326.0; 358.0 ];
       [ 163.0; 216.0; 242.0; 281.0; 312.0 ];
       [ 160.0; 207.0; 248.0; 288.0; 324.0 ];
       [ 142.0; 187.0; 234.0; 280.0; 316.0 ];
       [ 156.0; 203.0; 243.0; 283.0; 317.0 ];
       [ 157.0; 212.0; 259.0; 307.0; 336.0 ];
       [ 152.0; 203.0; 246.0; 286.0; 321.0 ];
       [ 154.0; 205.0; 253.0; 298.0; 334.0 ];
       [ 139.0; 190.0; 225.0; 267.0; 302.0 ];
       [ 146.0; 191.0; 229.0; 272.0; 302.0 ];
       [ 157.0; 211.0; 250.0; 285.0; 323.0 ];
       [ 132.0; 185.0; 237.0; 286.0; 331.0 ];
       [ 160.0; 207.0; 257.0; 303.0; 345.0 ];
       [ 169.0; 216.0; 261.0; 295.0; 333.0 ];
       [ 157.0; 205.0; 248.0; 289.0; 316.0];
       [ 137.0; 180.0; 219.0; 258.0; 291.0 ];
       [ 153.0; 200.0; 244.0; 286.0; 324.0]
    ]

// x data
let RatsXData:double[] = [| 8.0; 15.0; 22.0; 29.0; 36.0 |]

//CHart 1
let Points = [ for c in 0 .. 4 do
                  for r in 0..10 ->
                      RatsXData.[c], RatsHeightData.[r,c] ]
Chart.Point Points


//The model---------------------------------------------------
Rand.Restart(12347)
let N = Range(RatsHeightData.GetLength(0)).Named("N")
let T = Range(RatsHeightData.GetLength(1)).Named("T")
let alphaC =   Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("alphaC")
let alphaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("alphaTau")

let alpha = (Variable.ArrayInit N (fun _ -> Variable.GaussianFromMeanAndPrecision(alphaC, alphaTau)) ).Named("alpha")
let betaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("betaC")
let betaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("betaTau")

let beta = (Variable.ArrayInit N (fun _ -> Variable.GaussianFromMeanAndPrecision(betaC, betaTau)) ).Named("beta")
let tauC = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("tauC")
let x = Variable.Observed<double>(RatsXData, T).Named("x")
let xbar = Variable.Sum(x) / (float T.SizeAsInt) //CA
let y = Variable.Observed<double>(RatsHeightData, N, T).Named("y")

Variable.AssignVariableArray2D y N T (fun n t  -> Variable.GaussianFromMeanAndPrecision(alpha.[n] + (beta.[n] * (x.[t] - xbar)), tauC) )

let alpha0 = (alphaC - betaC * xbar).Named("alpha0")


// Initialise with the mean of the prior (needed for Gibbs to converge quickly)
alphaC.InitialiseTo(Gaussian.PointMass(0.0))
tauC.InitialiseTo(Gamma.PointMass(1.0))
alphaTau.InitialiseTo(Gamma.PointMass(1.0))
betaTau.InitialiseTo(Gamma.PointMass(1.0))

 // Inference engine
let ie = InferenceEngine()
ie.ShowFactorGraph<-true //false

let betaCMarg = ie.Infer<Gaussian>(betaC)
(* val betaCMarg : Gaussian = Gaussian(6.186, 0.01315) *)

let alpha0Marg = ie.Infer<Gaussian>(alpha0)
(* val alpha0Marg : Gaussian = Gaussian(106.4, 13.99) *)
let tauCMarg = ie.Infer<Gamma>(tauC)
(* val tauCMarg : Gamma = Gamma(42.26, 0.0006696)[mean=0.0283] *)

printfn "alpha0 = %A [sd=%s]"  (alpha0Marg)  (System.Math.Sqrt(alpha0Marg.GetVariance()).ToString("g4"))
(* alpha0 = Gaussian(106.4, 13.99) [sd=3.74] *)

printfn "tauC = %A" tauCMarg
(* tauC = Gamma(42.26, 0.0006696)[mean=0.0283] *)


//===========================================
// Charting
//---------
#I @"C:\Users\inter\OneDrive\Projects(Comp)\Dev_2018\Infer.NET_2018\packages"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open MathNet.Numerics
//open MathNet.Numerics.Distributions  //CA name conflict
//open MathNet.Numerics.Statistics

#load "FSharp.Charting.fsx"
open FSharp.Charting

let getValues (histogram:Histogram) =
    let bucketWidth = Math.Abs(histogram.LowerBound - histogram.UpperBound) / (float histogram.BucketCount)
    [0..(histogram.BucketCount-1)]
    |> Seq.map (fun i -> (histogram.Item(i).LowerBound + histogram.Item(i).UpperBound)/2.0, histogram.Item(i).Count)

MathNet.Numerics.Distributions.

//Normal------------------------------------
let dist = new Normal(0.0, 1.0)
let samples = dist.Samples() |> Seq.take 10000 |> Seq.toList
let histogram = new Histogram(samples, 35)

Chart.Column (getValues histogram)

//------------------------------------------
let dist2 = new MathNet.Numerics.Distributions.Bernoulli(0.5)
let samples2 = dist.Samples() |> Seq.take 10000 |> Seq.map (fun x-> float x) |>Seq.toList
let histogram2 = new Histogram(samples2, 35)

Chart.Column (getValues histogram2)

(*
Consider using the .Net Core language services by setting `FSharp.fsacRuntime` to `netcore`
.netcore location:
C:\Users\inter\.nuget\packages\runtime.win-x86.microsoft.netcore.app\2.1.0\runtimes\win-x86\lib\netcoreapp2.1
*)
