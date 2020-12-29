//Infer_Multinomial.fsx
// Converted to F# by Celso Axelrud on 12/29/2020 - rev 1.0
(*Refs:
    https://github.com/dotnet/infer
    https://dotnet.github.io/infer/userguide/The%20softmax%20factor.html
    https://github.com/dotnet/infer/blob/master/src/Tutorials/MultinomialRegression.cs

*)
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

open MathNet.Numerics
//open NumSharp


//Multinomial Regression
// For the multinomial regression model: generate synthetic data,
// infer the model parameters and calculate the RMSE between the true
// and mean inferred coefficients. 
//let engine = InferenceEngine()
//engine.Algorithm <- VariationalMessagePassing();

let numSamples = 1000
let numFeatures = 6
let numClasses = 4
let countPerSample = 10


let features = Array.zeroCreate<Vector> numSamples

let counts=Array.create numSamples (Array.zeroCreate<int> numClasses)

let coefficients=Array.zeroCreate<Vector> numClasses

let mean = Vector.Zero(numClasses)

Rand.Restart(1)

for i in 0..(numClasses - 1) do
    mean.[i] <- Rand.Normal()
    coefficients.[i] <- Vector.Zero(numFeatures)
    Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), coefficients.[i])

mean.[numClasses - 1] <- 0.0
coefficients.[numClasses - 1] <- Vector.Zero(numFeatures)

for i in 0..(numSamples - 1) do
    features.[i] <- Vector.Zero(numFeatures)
    Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), features.[i])
    let temp=Vector.FromArray([|for o in coefficients do yield o.Inner(features.[i])|])
    let p = MMath.Softmax(temp + mean)
    counts.[i] <- Rand.Multinomial(countPerSample, p)


Rand.Restart(DateTime.Now.Millisecond)

//let bPost:VectorGaussian[]=null
//let meanPost:Gaussian[]=null 

//MultinomialRegressionModel(features, counts, out bPost, out meanPost)
//Vector[] xObs, int[][] yObs, out VectorGaussian[] bPost, out Gaussian[] meanPost

let xObs=features
let yObs=counts

let C = yObs.[0].Length
let N = xObs.Length
let K = xObs.[0].Count;
let c = Range(C).Named("c")
let n = Range(N).Named("n")

// model
let B = (Variable.ArrayInit  
              c (fun c -> Variable.VectorGaussianFromMeanAndPrecision(  
                            Vector.Zero(K),  
                            PositiveDefiniteMatrix.Identity(K)))  ).Named("coefficients")

let m = (Variable.ArrayInit  
              c (fun c -> Variable.GaussianFromMeanAndPrecision(0.0,1.0 ))).Named("mean")

Variable.ConstrainEqualRandom(B.[C - 1],VectorGaussian.PointMass(Vector.Zero(K)))
Variable.ConstrainEqualRandom(m.[C - 1], Gaussian.PointMass(0.0))

let x = Variable.Array<Vector>(n)
x.ObservedValue <- xObs

let yData = Variable.Array<int>(Variable.Array<int>(c), n)
yData.ObservedValue <- yObs

let trialsCount = Variable.Array<int>(n)

trialsCount.ObservedValue<-[|for o in yObs do yield Seq.sum o|]

let g = Variable.Array<double>(Variable.Array<double>(c), n)

ForEach n {
            ForEach c {
                        g.[n].[c] <- Variable.InnerProduct(B.[c], x.[n]) + m.[c] 
            }
}

let p = Variable.Array<Vector>(n)

ForEach n {
            p.[n] <- Variable.Softmax(g.[n])
           }

ForEach n {
            yData.[n] <- Variable.Multinomial(trialsCount.[n], p.[n])
          }

 // inference
let ie = InferenceEngine(VariationalMessagePassing())
let bPost = ie.Infer<VectorGaussian[]>(B)
(*
val bPost : VectorGaussian [] =
  [|VectorGaussian(-0.4361 -0.1413 1.404 1.673 -0.6523 -0.2118, 0.001158   -0.0001594 1.438e-05  7.269e-05  1.587e-05  -3.213e-05)
                                                            -0.0001594 0.001179   -1.232e-05 -8.081e-05 3.247e-05  2.651e-05
                                                            1.438e-05  -1.232e-05 0.001084   2.006e-05  -1.564e-05 4.831e-05
                                                            7.269e-05  -8.081e-05 2.006e-05  0.001004   9.583e-05  4.184e-05
                                                            1.587e-05  3.247e-05  -1.564e-05 9.583e-05  0.001206   -0.000162
                                                            -3.213e-05 2.651e-05  4.831e-05  4.184e-05  -0.000162  0.001119  ;
    VectorGaussian(-0.6641 0.2584 1.153 0.9258 -1.332 0.7667, 0.0009084  -9.622e-05 -5.131e-05 8.831e-07  3.127e-06  8.825e-06)
                                                          -9.622e-05 0.0008629  2.645e-05  -1.281e-05 3.047e-06  1.628e-05
                                                          -5.131e-05 2.645e-05  0.0008575  4.027e-05  -2.135e-05 2.001e-07
                                                          8.831e-07  -1.281e-05 4.027e-05  0.0007964  5.861e-05  1.284e-05
                                                          3.127e-06  3.047e-06  -2.135e-05 5.861e-05  0.0008823  -0.000116
                                                          8.825e-06  1.628e-05  2.001e-07  1.284e-05  -0.000116  0.0008806;
    VectorGaussian(1.058 -1.085 -0.4151 0.3467 -0.4634 0.9807, 0.001236   -0.0003612 -0.000328  -0.0001084 -9.76e-07  0.0002683 )
                                                           -0.0003612 0.001069   0.0002832  6.558e-05  -3.763e-05 -0.0001707
                                                           -0.000328  0.0002832  0.001066   6.931e-05  -3.702e-05 -0.0002004
                                                           -0.0001084 6.558e-05  6.931e-05  0.0008146  4.797e-05  -1.597e-05
                                                           -9.76e-07  -3.763e-05 -3.702e-05 4.797e-05  0.000759   -4.814e-05
                                                           0.0002683  -0.0001707 -0.0002004 -1.597e-05 -4.814e-05 0.0009156 ;
    VectorGaussian.PointMass(0 0 0 0 0 0)|]
*)
let meanPost = ie.Infer<Gaussian[]>(m)
(*
val meanPost : Gaussian [] =
  [|Gaussian(-0.3993, 0.001021); Gaussian(0.3391, 0.0007578);
    Gaussian(0.8917, 0.0007456); Gaussian.PointMass(0)|]
*)

let bMeans=[|for o in bPost -> o.GetMean() |]
let bVars=[|for o in bPost -> o.GetVariance() |]
let mutable error=0.0


printfn "Coefficients:"
for i in 0..(numClasses-1) do
    error<- error + ( (bMeans.[i] - coefficients.[i]) |> Seq.sumBy (fun x -> x * 2.0) )
    //error <- error + (bMeans.[i] - coefficients.[i]) |> Seq.sumBy (fun x -> x * 2.0)
    printfn "Error:%f" error
    printfn "Class %i True     %O:" i coefficients.[i]
    printfn "Class %i Inferred %O:"  i bMeans.[i]

(*
Coefficients:
Error:-1.187990
Class 0 True     -0.4636 -0.1873 1.542 1.904 -0.8125 -0.1626:
Class 0 Inferred -0.4361 -0.1413 1.404 1.673 -0.6523 -0.2118:
Error:-1.433920
Class 1 True     -0.7679 0.2744 1.309 1.113 -1.488 0.7898:
Class 1 Inferred -0.6641 0.2584 1.153 0.9258 -1.332 0.7667:
Error:-1.639600
Class 2 True     1.047 -1.087 -0.3783 0.4571 -0.5071 0.9931:
Class 2 Inferred 1.058 -1.085 -0.4151 0.3467 -0.4634 0.9807:
Error:-1.639600
Class 3 True     0 0 0 0 0 0:
Class 3 Inferred 0 0 0 0 0 0:
*)

