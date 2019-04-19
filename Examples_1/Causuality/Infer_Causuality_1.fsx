//PLL_infer_7.fsx
//Causuality 
//==========

(*Refs:
    https://github.com/dotnet/infer/blob/master/src/Tutorials/CausalityExample.cs
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
//open Microsoft.ML.Probabilistic.Math

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
*)

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for Causuality - Learning causal relationships
//-----------------------------------------------------------------------------------
 (*
 Example of learning causal relationships from data using gates in Infer.NET.
 In this example, we consider two Boolean variables, A and B, and attempt to 
 distinguish whether A causes B or vice versa, through the use of data 
 with or without interventions on B.
*)
// Number of data points
let numberOfDataPoints = 20 

(*
Noise parameter - defines the true strength of the association between A and B
This ranges from 0.0 (meaning that A and B are equal) to 0.5 
(meaning that A and B are uncorrelated).
*)
let q = 0.1

(*
How we choose to set B in an intervention e.g. 0.5 is by a coin flip, 
This is a chosen parameter of our randomized study.
*)
let probBIntervention = 0.5

(*
Model definition:
Now we write the Infer.NET model to compare between A causing B and B causing A
 - in this example we only consider these two possibilities.
Gates are used to select between the two possibilities and to represent
perfect interventions. In Infer.NET gates are represented as stochastic if 
statements created using Variable.If() and Variable.IfNot().
*)

// Uniform prior over our two hypotheses 
// (True = A causes B, False = B causes A)
let AcausesB = Variable.Bernoulli(0.5)

// Range across the data
let N = Range(numberOfDataPoints)

// Set up array variables for the data
let A = Variable.Array<bool>(N).Named("A")
let B = Variable.Array<bool>(N).Named("B")
let doB = Variable.Array<bool>(N).Named("doB")

// Loop over the data points
ForEach N {
          // Intervention case - this is the same for either model
          //defined once here.
    If doB.[N] {
          // Given intervention B is selected at random 
          // using a known parameter e.g. 0.5.
        B.[N] <- Variable.Bernoulli probBIntervention
    }
}

// First model: A causes B
If AcausesB {
    // Loop over the data points
    ForEach N {
        // Draw A from uniform prior
        A.[N] <- Variable.Bernoulli 0.5
        // No intervention case for the A causes B model
        IfNot doB.[N] {
            B.[N] <- A.[N] <<>> Variable.Bernoulli q
            //B.[N] <- Variable.NotEqual (A.[N], Variable.Bernoulli q)
        }
    }
}

// Second model: B causes A
IfNot AcausesB {
    // Loop over the data points
    ForEach N {
        // No intervention case for the B causes A model
        IfNot doB.[N] {
            // Draw B from uniform prior
            B.[N] <- Variable.Bernoulli 0.5
        }
        // Set A to a noisy version of B
        A.[N] <- B.[N] <<>> Variable.Bernoulli q
        //A.[N] <- Variable.NotEqual (B.[N], Variable.Bernoulli q)
    }
}

// Inference
// Create an Infer.NET inference engine
let engine = InferenceEngine()
printfn "Causal inference using gates in Infer.NET"
printfn "========================================="
printfn "Data set of  %i  data points with noise %f" numberOfDataPoints q 


// Class to store the data
type Data = {   A:bool[]; // observations of A
                B:bool[]; // observations of B
                doB:bool[] // whether we intervened to set B
    }

// Generates data from the true model: A cause B
//N: Number of data points to generate</param>
// q:Noise (flip) probability
//doB:Whether to intervene or not</param>
//probBIntervention:Prob of choosing B=true when intervening
let GenerateFromTrueModel(N:int, q:double, doB:bool, probBIntervention:double)=
            // Create data object to fill with data.
            let d = { A = Array.zeroCreate N; B = Array.zeroCreate N; doB = Array.zeroCreate N }
            // Uniform prior on A
            let Aprior = Bernoulli(0.5)
            // Noise distribution
            let flipDist = Bernoulli(q)
            // Distribution over the values of B when we intervene 
            let interventionDist = Bernoulli(probBIntervention)

            // Loop over data
            for i in  0..N-1 do
                // Draw A from prior
                d.A.[i] <- Aprior.Sample()
                // Whether we intervened on B 
                // This is currently the same for all data points - but could easily be modified.
                d.doB.[i] <- doB
                if not d.doB.[i] then 
                    // We are not intervening so use the causal model i.e.
                    // make B a noisy version of A - flipping it with probability q
                    d.B.[i] <- d.A.[i] <> flipDist.Sample();
                else
                    // We are intervening - setting B according to a coin flip
                    d.B.[i] <- interventionDist.Sample()
            d

// Data without interventions
// Generate data set
let dataWithoutInterventions = GenerateFromTrueModel(numberOfDataPoints, q, false, probBIntervention)

// Attach the data without interventions
A.ObservedValue <- dataWithoutInterventions.A
B.ObservedValue <- dataWithoutInterventions.B
doB.ObservedValue <- dataWithoutInterventions.doB

// Infer probability that A causes B (rather than B causes A)
let AcausesBdist = engine.Infer<Bernoulli>(AcausesB)
printfn "P(A causes B), without interventions=%f"  (AcausesBdist.GetProbTrue())
(* P(A causes B), without interventions=0.500000 *)

// Data WITH interventions 
// Number of inference runs to average over (each with a different generated data set)
let numberOfRuns = 10
printfn "Executing %i  runs with interventions:" numberOfRuns
(*Executing 10  runs with interventions:*)
let mutable tot = 0.0

for i in  0..numberOfRuns-1 do
    // Generate data with interventions
    let dataWithInterventions = GenerateFromTrueModel(numberOfDataPoints, q, true, probBIntervention)
    // Attach the data with interventions (this replaces any previously attached data)
    A.ObservedValue <- dataWithInterventions.A
    B.ObservedValue <- dataWithInterventions.B
    doB.ObservedValue <- dataWithInterventions.doB
    // Infer probability that A causes B (rather than B causes A)
    let AcausesBdist2 = engine.Infer<Bernoulli>(AcausesB)
    let r1=AcausesBdist2.GetProbTrue()
    tot <- tot + r1
    printfn "%i. P(A causes B)=%f" (i + 1) r1
    printfn "Average P(A causes B), with interventions= %f" (tot/(float numberOfRuns))

(* 
1. P(A causes B)=0.999963
Average P(A causes B), with interventions= 0.099996
2. P(A causes B)=0.999996
Average P(A causes B), with interventions= 0.199996
3. P(A causes B)=0.999996
Average P(A causes B), with interventions= 0.299996
4. P(A causes B)=0.999963
Average P(A causes B), with interventions= 0.399992
5. P(A causes B)=0.974039
Average P(A causes B), with interventions= 0.497396
6. P(A causes B)=0.999671
Average P(A causes B), with interventions= 0.597363
7. P(A causes B)=0.999996
Average P(A causes B), with interventions= 0.697362
8. P(A causes B)=1.000000
Average P(A causes B), with interventions= 0.797362
9. P(A causes B)=1.000000
Average P(A causes B), with interventions= 0.897362
10. P(A causes B)=0.999671
Average P(A causes B), with interventions= 0.997329
*)

