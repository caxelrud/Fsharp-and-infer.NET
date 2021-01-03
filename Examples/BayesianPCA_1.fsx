//BayesianPCA_1.fsx
//PPCA 
//==========

// Converted to F# by Celso Axelrud on 1/3/2020 - rev 1.0
(*Refs:
    https://github.com/dotnet/infer
    https://dotnet.github.io/infer/userguide/Bayesian%20PCA%20and%20Factor%20Analysis.html
*)

#I @"C:\Users\inter\OneDrive\_myWork\Research2020\Infernet_2020"
#r "Microsoft.ML.Probabilistic.Compiler.dll"
#r "Microsoft.ML.Probabilistic.dll"
#r "Microsoft.ML.Probabilistic.FSharp.dll"
#r "MathNet.Numerics.dll"


open System
open Microsoft.ML.Probabilistic.Collections
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.Utilities
open Microsoft.ML.Probabilistic.Algorithms

open System.Collections.Generic
//open MathNet.Numerics

// True W. Inference will find a different basis
let trueW:double[,] =
        array2D
            [|
              [| -0.30; 0.40; 0.20; -0.15; 0.20; -0.25; -0.50; -0.10; -0.25; 0.10 |];
              [| -0.10; -0.20; 0.40; 0.50; 0.15; -0.35; 0.05; 0.20; 0.20; -0.15 |];
              [| 0.15; 0.05; 0.15; -0.10; -0.15; 0.25; -0.10; 0.15; -0.30; -0.55 |];
            |]
// True bias
let trueMu:double[] = [| -0.95; 0.75; -0.20; 0.20; 0.30; -0.35; 0.65; 0.20; 0.25; 0.40 |]
// True observation noise
let truePi:double[] = [| 8.0; 9.0; 10.0; 11.0; 10.0; 9.0; 8.0; 9.0; 10.0; 11.0 |]

// Generate data from the true model
// numObs:Number of observations
let generateData(numObs:int)=
    let numComp = trueW.GetLength(0)
    let numFeat = trueW.GetLength(1)
    let data = Array2D.zeroCreate<double> numObs numFeat
    let WMat = Matrix(trueW)
    let z = Vector.Zero(numComp)
    for i in 0..(numObs-1) do
                // Sample scores from standard Gaussian
                for j in 0..(numComp-1) do
                    z.[j] <- Gaussian.Sample(0.0, 1.0)
                // Mix the components with the true mixture matrix
                let t = z * WMat
                for j in 0..(numFeat-1) do
                    // Add in the bias
                    let u = t.[j] + trueMu.[j]
                    // ... and the noise
                    data.[i, j] <- Gaussian.Sample(u, truePi.[j])
    data



// Run a Bayesian PCA example
// "A Bayesian Principal Components Analysis example"

// Set a stable random number seed for repeatable runs
Rand.Restart(12347);
let Data = generateData(1000)

// Inference engine
let engine = InferenceEngine(VariationalMessagePassing())

// Model variables
let observationCount = Variable.New<int>().Named("observationCount")
let featureCount = Variable.New<int>().Named("featureCount")
let componentCount = Variable.New<int>().Named("componentCount")

let observation = Range(observationCount).Named("observation")
let feature = Range(featureCount).Named("feature")
let component = Range(componentCount).Named("component")

let data = Variable.Array<double>(observation, feature).Named("data")

//priors
let priorAlpha = Variable.New<Gamma>().Named("priorAlpha")
let priorMu = Variable.New<Gaussian>().Named("priorMu")
let priorPi = Variable.New<Gamma>().Named("priorPi")

// Mixing matrix. Each row is drawn from a Gaussian with zero mean and
// a precision which will be learnt. This is a form of Automatic
// Relevance Determination (ARD). The larger the precisions become, the
// less important that row in the mixing matrix is in explaining the data
let alpha = Variable.Array<float>(component).Named("alpha")
let W = Variable.Array<float>(component, feature).Named("W")

ForEach component {
    alpha.[component] <- Variable<double>.Random<Gamma>(priorAlpha)    
    }

ForEach feature {
        W.[component, feature] <- Variable.GaussianFromMeanAndPrecision(Variable.Constant(0.0), alpha.[component])
        }

// Initialize the W marginal to break symmetry
let initW = Variable.Array<Gaussian>(component, feature).Named("initW")

observationCount.ObservedValue <- Data.GetLength(0)
featureCount.ObservedValue <- Data.GetLength(1);

// Set the data
data.ObservedValue <- Data

// Set the dimensions
componentCount.ObservedValue <- 6

// Set the priors
priorMu.ObservedValue <- Gaussian.FromMeanAndPrecision(0.0, 0.01)
priorPi.ObservedValue <- Gamma.FromShapeAndRate(2.0, 2.0)
priorAlpha.ObservedValue <- Gamma.FromShapeAndRate(2.0, 2.0)

// Set the initialization
initW.ObservedValue <- Array2D.init componentCount.ObservedValue featureCount.ObservedValue (fun i j -> Gaussian.FromMeanAndVariance(Rand.Normal(), 1.0))

//data
W.[component, feature].InitialiseTo(initW.[component, feature])

// Latent variables are drawn from a standard Gaussian
let Z = Variable.Array<double>(observation, component).Named("Z")

ForEach observation {
    ForEach component {
        Z.[observation, component] <- Variable.GaussianFromMeanAndPrecision(0.0, 1.0)    }
}

// Multiply the latent variables with the mixing matrix...
let T = Variable.MatrixMultiply(Z, W).Named("T");

// ... add in a bias ...
let mu = Variable.Array<double>(feature).Named("mu");
ForEach feature {
    mu.[feature] <- Variable<double>.Random<Gaussian>(priorMu)
}

let U = Variable.Array<double>(observation, feature).Named("U")
U.[observation, feature] <- T.[observation, feature] + mu.[feature];

// ... and add in some observation noise ...
let pi = Variable.Array<double>(feature).Named("pi")

ForEach feature {
    pi.[feature] <- Variable.Random(priorPi)
}

// ... to give the likelihood of observing the data
data.[observation, feature] <- Variable.GaussianFromMeanAndPrecision(U.[observation, feature], pi.[feature])

// Infer the marginals
engine.NumberOfIterations <- 200
let inferredW = engine.Infer<IArray2D<Gaussian>>(W)
let inferredMu = engine.Infer<IReadOnlyList<Gaussian>>(mu)
let inferredPi = engine.Infer<IReadOnlyList<Gamma>>(pi);


let meanAbsoluteRowMeans( matrix:IArray2D<Gaussian>)=
    let mam = Array.zeroCreate (matrix.GetLength(0))
    let mult:double = 1.0 / (double (matrix.GetLength(1)))
    for i in 0..(matrix.GetLength(0)-1) do
        let mutable sum:double = 0.0
        for j in 0..(matrix.GetLength(1)-1) do
            sum<- sum + System.Math.Abs(matrix.[i, j].GetMean())
            mam.[i] <- mult * sum;
    mam

// Print out the results
printfn "Inferred W: %A" inferredW 
//Inferred W: seq
//  [Gaussian(-0.1069, 0.0001271); Gaussian(0.4353, 0.0001099);
//   Gaussian(-0.154, 9.439e-05); Gaussian(-0.4721, 9.78e-05); ...]
printfn "Mean absolute means of rows in W: %A" (meanAbsoluteRowMeans(inferredW))
//Mean absolute means of rows in W: [|0.2366793441; 0.2513085693; 0.01686112128; 0.02435795515; 0.1742009515;0.0004694614013|]
printfn "    True bias: %A " trueMu
//True bias: [|-0.95; 0.75; -0.2; 0.2; 0.3; -0.35; 0.65; 0.2; 0.25; 0.4|]
printfn "Inferred bias: %A" [for d in inferredMu -> d.GetMean()]
//Inferred bias: [-0.9367105521; 0.7236915574; -0.2163834273; 0.2075330278; 0.3051779293;-0.3648300348; 0.6517707303; 0.1790643469; 0.2541546284; 0.4218541924]
printfn "    True noise: %A" truePi
//True noise: [|8.0; 9.0; 10.0; 11.0; 10.0; 9.0; 8.0; 9.0; 10.0; 11.0|]
printfn "Inferred noise: %A" [for d in inferredPi -> d.GetMean()]
//Inferred noise: [7.925228477; 9.163346054; 10.67342098; 10.30095634; 10.1938076; 8.540025391;7.541301429; 7.973112052; 10.50435726; 9.204049665]
