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
//#r "FSharp.Core.dll"

open System
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.Algorithms

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

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for CyclingTime 1
// No Object-Oriented
//-----------------------------------------------------------------------------------

// [1] The model
let averageTime =  Variable.GaussianFromMeanAndPrecision(15.0, 0.01).Named("AverageTime") // precision:reciprocal of the variance (1/sigma^2)
let trafficNoise = Variable.GammaFromShapeAndScale(2.0, 0.5).Named("trafficNoise") 
//let trafficNoise1 = Variable.GammaFromShapeAndRate(2.0,2.0)
let travelTimeMonday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise).Named("TravelTimeMon")
let travelTimeTuesday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise).Named("TravelTimeTue")
let travelTimeWednesday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise).Named("TravelTimeWed")

// [2] Train the model
travelTimeMonday.ObservedValue <- 13.0
travelTimeTuesday.ObservedValue <- 17.0
travelTimeWednesday.ObservedValue <- 16.0

let engine = InferenceEngine()
engine.ShowFactorGraph<-false 

let averageTimePosterior = engine.Infer<Gaussian>(averageTime)
let trafficNoisePosterior = engine.Infer<Gamma>(trafficNoise)

printfn "averageTimePosterior (mean,variance): %A" averageTimePosterior
printfn "trafficNoisePosterior: %A" trafficNoisePosterior
(*
averageTimePosterior: Gaussian(15.33, 1.32)
trafficNoisePosterior: Gamma(2.242, 0.2445)[mean=0.5482]  
*)

let dist2 = MathNet.Numerics.Distributions.Gamma(trafficNoisePosterior.Shape,trafficNoisePosterior.Rate)
let samples2 = dist2.Samples() |> Seq.take 10000 |> Seq.map (fun x-> float x) |>Seq.toList
let histogram2 = Statistics.Histogram(samples2, 35)
Chart.Column (getValues histogram2)


// [3] Add a prediction variable and retrain the model
let tomorrowsTime = Variable.GaussianFromMeanAndPrecision(averageTime,trafficNoise).Named("TomorrowsTime")
engine.ShowFactorGraph<-false
let tomorrowsTimeDist = engine.Infer<Gaussian>(tomorrowsTime)
let tomorrowsMean = tomorrowsTimeDist.GetMean()
let tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance())
printfn "Tomorrows predicted time: %f plus or minus %f" tomorrowsMean tomorrowsStdDev
(*
Tomorrows predicted time: 15.326309 plus or minus 2.147767
*)

// You can also ask other questions of the model 
let probTripTakesLessThan18Minutes = engine.Infer<Bernoulli>(tomorrowsTime << 18.0).GetProbTrue()
printfn "Probability that the trip takes less than 18 min: %f" probTripTakesLessThan18Minutes
(*
Probability that the trip takes less than 18 min: 0.893410
*)

//INPUT RETRACTION
//----------------
//Add observed value for tomorrowsTime (CA)
tomorrowsTime.ObservedValue <- tomorrowsMean
// Clear TimeWednesday
travelTimeWednesday.ClearObservedValue()
let travelTimeWednesdayDist = engine.Infer<Gaussian>(travelTimeWednesday)
//val travelTimeWednesdayDist : Gaussian = Gaussian(15.11, 4.301)


//-----------------------------------------------------------------------------------
// Infer.NET: F# script for CyclingTime 2
// Object-oriented
//-----------------------------------------------------------------------------------
type ModelData(mean:Gaussian,precision:Gamma)=
  member this.AverageTimeDist=mean
  member this.TrafficNoiseDist=precision

type CyclistBase()=  
  let mutable ie:InferenceEngine=null
  let mutable averageTime:Variable<double>=null
  let mutable trafficNoise:Variable<double>=null
  let mutable averageTimePrior:Variable<Gaussian>=null
  let mutable trafficNoisePrior:Variable<Gamma>=null

  member this.IE with get()=ie and set (value) = ie <- value
  member this.AverageTime=averageTime
  member this.TrafficNoise=trafficNoise
  member this.AverageTimePrior=averageTimePrior
  member this.TrafficNoisePrior=trafficNoisePrior

  abstract member CreateModel: unit->unit
  default this.CreateModel()=  
    averageTimePrior <- Variable.New<Gaussian>() 
    averageTime <- Variable<double>.Random<Gaussian>(averageTimePrior)
    trafficNoisePrior <- Variable.New<Gamma>()
    trafficNoise <- Variable<double>.Random<Gamma>(trafficNoisePrior) 
    if ie=null then
      ie <- InferenceEngine()
    ()

  abstract member SetModelData: ModelData->unit
  default this.SetModelData(priors:ModelData )=  
    averageTimePrior.ObservedValue <- priors.AverageTimeDist
    trafficNoisePrior.ObservedValue <- priors.TrafficNoiseDist
    ()

type CyclistTraining()=
  inherit CyclistBase()

  let  mutable travelTimes:VariableArray<double>=null
  let  mutable numTrips:Variable<int>=null

  member this.TravelTimes=travelTimes
  member this.NumTrips=numTrips

  override this.CreateModel()=  
    base.CreateModel()
    numTrips <- Variable.New<int>()
    let tripRange = Range(numTrips)
    travelTimes <- Variable.Array<double>(tripRange)
    //do printfn "%A" averageTime
    //do printfn "%A" this.IE
    do Variable.ForeachBlock tripRange (fun t-> travelTimes.[t] <- Variable.GaussianFromMeanAndPrecision(this.AverageTime, this.TrafficNoise))

  member this.InferModelData(trainingData:double[])=
            numTrips.ObservedValue <- trainingData.Length
            travelTimes.ObservedValue <- trainingData
            let r1 = this.IE.Infer<Gaussian>(this.AverageTime)
            let r2 = this.IE.Infer<Gamma>(this.TrafficNoise);
            let posteriors = ModelData(r1,r2)
            posteriors

type CyclistPrediction()=
  inherit CyclistBase()
  let mutable tomorrowsTimeDist:Gaussian=Gaussian()
  let mutable tomorrowsTime:Variable<double>=null
  member this.TomorrowsTimeDist=tomorrowsTimeDist
  member this.TomorrowsTime=tomorrowsTime

  override this.CreateModel()=  
    base.CreateModel()
    tomorrowsTime <- Variable.GaussianFromMeanAndPrecision(this.AverageTime, this.TrafficNoise)
    tomorrowsTimeDist <- Gaussian()
    ()

  member this.InferTomorrowsTime()=
    let tomorrowsTimeDist = this.IE.Infer<Gaussian>(tomorrowsTime)
    tomorrowsTimeDist

  member this.InferProbabilityTimeLessThan(time:double)=
            this.IE.Infer<Bernoulli>(tomorrowsTime << time)


//Run
let trainingData = [| 13.0; 17.0; 16.0; 12.0; 13.0; 12.0; 14.0; 18.0; 16.0; 16.0 |]
let initPriors = ModelData(Gaussian.FromMeanAndPrecision(1.0, 0.01),Gamma.FromShapeAndScale(2.0, 0.5))

// Train the model
let cyclistTraining = CyclistTraining()
cyclistTraining.CreateModel()

cyclistTraining.SetModelData(initPriors)
let posteriors1 = cyclistTraining.InferModelData(trainingData)
printfn "Average travel time = %A "  posteriors1.AverageTimeDist
printfn "Traffic noise = %A" posteriors1.TrafficNoiseDist
(*
Average travel time = Gaussian(14.65, 0.4459)
Traffic noise = Gamma(5.33, 0.05399)[mean=0.2878]  
*)
let cyclistPrediction = CyclistPrediction()
cyclistPrediction.CreateModel()
cyclistPrediction.SetModelData(posteriors1)
let tomorrowsTimeDist = cyclistPrediction.InferTomorrowsTime()
let tomorrowsMean = tomorrowsTimeDist.GetMean()
let tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance())
printfn "Tomorrows average time: %f" tomorrowsMean
printfn "Tomorrows standard deviation: %f" tomorrowsStdDev
printfn "Probability that tomorrow's time is < 18 min: %A" (cyclistPrediction.InferProbabilityTimeLessThan(18.0))
(*
Tomorrows average time: 14.651578
Tomorrows standard deviation: 2.173326
Probability that tomorrow's time is < 18 min: Bernoulli(0.9383)  
*)


//BAYESIAN TRAINING (ONLINE LEARNING)
// Second round of training
let trainingData2 = [|17.0; 19.0; 18.0; 21.0; 15.0 |]
cyclistTraining.SetModelData(posteriors1)
let posteriors2 = cyclistTraining.InferModelData(trainingData2)
printfn "Average travel time = %A" posteriors2.AverageTimeDist
printfn "Traffic noise = %A"  posteriors2.TrafficNoiseDist
(*
Average travel time = Gaussian(15.61, 0.3409)
Traffic noise = Gamma(7.244, 0.02498)[mean=0.1809]  
*)

// Predictions based on two rounds of training
cyclistPrediction.SetModelData(posteriors2)
let tomorrowsTimeDist = cyclistPrediction.InferTomorrowsTime()
let tomorrowsMean = tomorrowsTimeDist.GetMean()
let tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance())
printfn "Tomorrows average time: %f" tomorrowsMean
printfn "Tomorrows standard deviation: %f" tomorrowsStdDev
printfn "Probability that tomorrow's time is < 18 min: %A" (cyclistPrediction.InferProbabilityTimeLessThan(18.0))
(*
Tomorrows average time: 15.608462
Tomorrows standard deviation: 2.598555
Probability that tomorrow's time is < 18 min: Bernoulli(0.8213)
*)

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for CyclingTime 3
// Mixing Models
//-----------------------------------------------------------------------------------

type ModelDataMixed(averageTimeDist:Gaussian[],trafficNoiseDist:Gamma[],mixingDist:Dirichlet)=
  member this.AverageTimeDist=averageTimeDist
  member this.TrafficNoiseDist=trafficNoiseDist
  member this.MixingDist=mixingDist

type CyclistMixedBase()=  
  let mutable ie:InferenceEngine=null
  let mutable averageTime:VariableArray<double>=null
  let mutable trafficNoise:VariableArray<double>=null
  let mutable averageTimePriors:VariableArray<Gaussian>=null 
  let mutable trafficNoisePriors:VariableArray<Gamma>=null

  let mutable mixingPrior:Variable<Dirichlet>=null
  let mutable mixingCoefficients:Variable<Vector>=null
  let numComponents=2

  member this.IE=ie
  member this.AverageTime=averageTime
  member this.TrafficNoise=trafficNoise
  member this.AverageTimePriors=averageTimePriors
  member this.TrafficNoisePriors=trafficNoisePriors

  member this.NumComponents=numComponents
  member this.MixingPrior=mixingPrior
  member this.MixingCoefficients=mixingCoefficients

  abstract member CreateModel: unit->unit
  default this.CreateModel()=
    let componentRange = Range(numComponents)

    averageTimePriors<-Variable.Array<Gaussian>(componentRange) 
    averageTime<-Variable.Array<double>(componentRange)
    trafficNoisePriors <- Variable.Array<Gamma>(componentRange)
    trafficNoise<-Variable.Array<double>(componentRange)
    
    ForEach componentRange {
      averageTime.[componentRange] <- Variable<double>.Random<Gaussian>(averageTimePriors.[componentRange])    
      trafficNoise.[componentRange] <- Variable<double>.Random<Gamma>(trafficNoisePriors.[componentRange])
      }

    mixingPrior<-Variable.New<Dirichlet>() 
    mixingCoefficients<-Variable<Vector>.Random<Dirichlet>(mixingPrior)
    mixingCoefficients.SetValueRange(componentRange)

    ie <- InferenceEngine(VariationalMessagePassing())
    ie.ShowProgress <- false
  
  abstract member SetModelData: ModelDataMixed->unit
  default this.SetModelData(modelData:ModelDataMixed)=  
    averageTimePriors.ObservedValue <-  modelData.AverageTimeDist
    trafficNoisePriors.ObservedValue <- modelData.TrafficNoiseDist
    mixingPrior.ObservedValue <- modelData.MixingDist
    ()

type CyclistMixedTraining() as this=
  inherit CyclistMixedBase()

  let mutable travelTimes:VariableArray<double>=null
  let mutable componentIndices:VariableArray<int>=null
  let mutable numTrips:Variable<int>=null

  member this.NumTrips=numTrips
  member this.TravelTimes=travelTimes
  member this.ComponentIndices=componentIndices

  override this.CreateModel()=  
    base.CreateModel()
    numTrips <- Variable.New<int>()
    let tripRange = Range(numTrips)
    travelTimes <- Variable.Array<double>(tripRange)
    componentIndices <- Variable.Array<int>(tripRange)
    ForEach tripRange {
                  componentIndices.[tripRange] <- Variable.Discrete(this.MixingCoefficients)
                  Switch componentIndices.[tripRange] {
                                                          travelTimes.[tripRange].SetTo(
                                                              Variable.GaussianFromMeanAndPrecision(this.AverageTime.[componentIndices.[tripRange]],
                                                                  this.TrafficNoise.[componentIndices.[tripRange]]))
                   }
        }
    ()
  member this.InferModelData(trainingData:double[])=
      numTrips.ObservedValue <- trainingData.Length
      travelTimes.ObservedValue <- trainingData
      let r1 = this.IE.Infer<Gaussian[]>(this.AverageTime)
      let r2 = this.IE.Infer<Gamma[]>(this.TrafficNoise);
      let r3 = base.IE.Infer<Dirichlet>(this.MixingCoefficients)
      let posteriors = ModelDataMixed(r1,r2,r3)     
      posteriors


type CyclistMixedPrediction()=
  inherit CyclistMixedBase()

  let mutable tomorrowsTimeDist:Gaussian=Gaussian()
  let mutable tomorrowsTime:Variable<double>=null

  member this.TomorrowsTimeDist=tomorrowsTimeDist
  member this.TomorrowsTime=tomorrowsTime

  override this.CreateModel()=  
    base.CreateModel()
    let componentIndex = Variable.Discrete(this.MixingCoefficients)
    tomorrowsTime <- Variable.New<double>()
    Switch componentIndex {
                tomorrowsTime.SetTo(
                      Variable.GaussianFromMeanAndPrecision(
                        this.AverageTime.[componentIndex],
                        this.TrafficNoise.[componentIndex]))
            }
  member this.InferTomorrowsTime()=
            let tomorrowsTimeDist = this.IE.Infer<Gaussian>(tomorrowsTime)
            tomorrowsTimeDist


//Run
let trainingData = [| 13.0; 17.0; 16.0; 12.0; 13.0; 12.0; 14.0; 18.0; 16.0; 16.0;27.0; 32.0 |]
let AverageTimeDist = [|Gaussian(15.0, 100.0); Gaussian(30.0, 100.0)|] 
let TrafficNoiseDist = [| Gamma(2.0, 0.5); Gamma(2.0, 0.5) |]
let MixingDist = Dirichlet(1.0, 1.0)

let initPriors = ModelDataMixed(AverageTimeDist,TrafficNoiseDist,MixingDist)

// Train the model
let cyclistMixedTraining = CyclistMixedTraining()
cyclistMixedTraining.CreateModel()
cyclistMixedTraining.SetModelData(initPriors)

let posteriors = cyclistMixedTraining.InferModelData(trainingData)
printfn "Average travel time distribution 1 = %A "  posteriors.AverageTimeDist.[0]
printfn "Average travel time distribution 2 = %A "  posteriors.AverageTimeDist.[1]
printfn "Traffic noise distribution 1 = %A" posteriors.TrafficNoiseDist.[0]
printfn "Traffic noise distribution 2 = %A" posteriors.TrafficNoiseDist.[1]
printfn "Mixing coefficient distribution = %A " posteriors.MixingDist
(*
Average travel time distribution 1 = Gaussian(14.7, 0.3533)
Average travel time distribution 2 = Gaussian(29.51, 1.618)
Traffic noise distribution 1 = Gamma(7, 0.0403)[mean=0.2821]
Traffic noise distribution 2 = Gamma(3, 0.1013)[mean=0.304]
Mixing coefficient distribution = Dirichlet(11 3)
*)

let cyclistMixedPrediction = CyclistMixedPrediction()
cyclistMixedPrediction.CreateModel()
cyclistMixedPrediction.SetModelData(posteriors)
let tomorrowsTime = cyclistMixedPrediction.InferTomorrowsTime()
let tomorrowsMean = tomorrowsTime.GetMean()
let tomorrowsStdDev = Math.Sqrt(tomorrowsTime.GetVariance())
printfn "Tomorrows average time: %f" tomorrowsMean
printfn "Tomorrows standard deviation: %f" tomorrowsStdDev
(*
Tomorrows average time: 17.033339
Tomorrows standard deviation: 5.711589
*)

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for CyclingTime 4
// Model Selection
//-----------------------------------------------------------------------------------

type CyclistWithEvidence()=
  inherit CyclistTraining()

  let mutable evidence:Variable<bool>=null
  member this.Evidence=evidence

  override this.CreateModel()=
    evidence <- Variable.Bernoulli(0.5)
    let mutable X=false
    let _=Variable.IfBlock evidence (fun (_:Variable<bool>) -> X<-true)
    //if X then base.CreateModel()
    base.CreateModel()
    ()

  member this.InferEvidence(trainingData:double[])=
    let posteriors = base.InferModelData(trainingData)
    let logEvidence = this.IE.Infer<Bernoulli>(evidence).LogOdds
    logEvidence

type CyclistMixedWithEvidence()=
  inherit CyclistMixedTraining()

  let mutable evidence:Variable<bool>=null
  member this.Evidence=evidence

  override this.CreateModel()=
    evidence <- Variable.Bernoulli(0.5)
    let mutable X=false
    let _=Variable.IfBlock evidence (fun (_:Variable<bool>) -> X<-true)
    //if X then base.CreateModel()
    base.CreateModel()
    ()

  member this.InferEvidence(trainingData:double[])=
    let posteriors = base.InferModelData(trainingData)
    let logEvidence = this.IE.Infer<Bernoulli>(evidence).LogOdds
    logEvidence

//Run
let trainingData4 = [| 13.0; 17.0; 16.0; 12.0; 13.0; 12.0; 14.0; 18.0; 16.0; 16.0;27.0; 32.0 |]
let initPriors4 = ModelData(Gaussian.FromMeanAndPrecision(15.0, 0.01),Gamma.FromShapeAndScale(2.0, 0.5));
let cyclistWithEvidence = CyclistWithEvidence()

cyclistWithEvidence.CreateModel()

//-----------
cyclistWithEvidence.SetModelData(initPriors4)
//cyclistWithEvidence.AverageTimePrior.ObservedValue <- initPriors4.AverageTimeDist //AverageTimePrior no an instance
//cyclistWithEvidence.TrafficNoisePrior.ObservedValue <- initPriors4.TrafficNoiseDist
//------------

let logEvidence = cyclistWithEvidence.InferEvidence(trainingData4)

let initPriorsMixed=ModelDataMixed(
                          [| Gaussian(15.0, 100.0); Gaussian(30.0, 100.0)|],
                          [| Gamma(2.0, 0.5); Gamma(2.0, 0.5) |],
                          Dirichlet(1.0, 1.0)
                )

let cyclistMixedWithEvidence = CyclistMixedWithEvidence()
cyclistMixedWithEvidence.CreateModel()

cyclistMixedWithEvidence.SetModelData(initPriorsMixed);

let logEvidenceMixed = cyclistMixedWithEvidence.InferEvidence(trainingData4)

printfn "Log evidence for single Gaussian: %f" logEvidence
printfn "Log evidence for mixture of two Gaussians: %f" logEvidenceMixed

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for CyclingTime 5
// Two Cyclist
//-----------------------------------------------------------------------------------

type TwoCyclistsTraining()=
  let cyclist1 = CyclistTraining()
  do cyclist1.CreateModel()
  let cyclist2 = CyclistTraining()
  do cyclist2.CreateModel()

  member this.Cyclist1=cyclist1
  member this.Cyclist2=cyclist2

  member this.SetModelData(modelData:ModelData)=
    cyclist1.SetModelData(modelData)
    cyclist2.SetModelData(modelData)

  member this.InferModelData(trainingData1:double[],trainingData2:double[])=
    let posteriors = Array.zeroCreate<ModelData> 2
    posteriors.[0] <- cyclist1.InferModelData(trainingData1)
    posteriors.[1] <- cyclist2.InferModelData(trainingData2)
    posteriors


type TwoCyclistsPrediction()=
  let CommonEngine = InferenceEngine()
  let cyclist1 = CyclistPrediction()
  do cyclist1.IE<-CommonEngine
  do cyclist1.CreateModel()
  let cyclist2 = CyclistPrediction()
  do cyclist2.IE<-CommonEngine
  do cyclist2.CreateModel()
  let TimeDifference = cyclist1.TomorrowsTime - cyclist2.TomorrowsTime
  let Cyclist1IsFaster = cyclist1.TomorrowsTime << cyclist2.TomorrowsTime

  member this.SetModelData(modelData:ModelData[])=
    cyclist1.SetModelData(modelData.[0]);
    cyclist2.SetModelData(modelData.[1]);

  member  this.InferTomorrowsTime()=
    let tomorrowsTime = Array.zeroCreate<Gaussian> 2
    tomorrowsTime.[0] <- cyclist1.InferTomorrowsTime()
    tomorrowsTime.[1] <- cyclist2.InferTomorrowsTime()
    tomorrowsTime

  member this.InferTimeDifference()=
    CommonEngine.Infer<Gaussian>(TimeDifference)

  member this.InferCyclist1IsFaster()=
    CommonEngine.Infer<Bernoulli>(Cyclist1IsFaster)

//Run
let trainingData5_1 = [| 13.0; 17.0; 16.0; 12.0; 13.0; 12.0; 14.0; 18.0; 16.0; 16.0; 27.0; 32.0 |]
let trainingData5_2 = [| 16.0; 18.0; 21.0; 15.0; 17.0; 22.0; 28.0; 16.0; 19.0; 33.0; 20.0; 31.0 |]

let initPriors5 = ModelData(Gaussian.FromMeanAndPrecision(15.0, 0.01),Gamma.FromShapeAndScale(2.0, 0.5))

// Train the model
let cyclistsTraining = TwoCyclistsTraining()
//cyclistsTraining.CreateModel()
cyclistsTraining.SetModelData(initPriors5)

let posteriors5 = cyclistsTraining.InferModelData(trainingData5_1, trainingData5_2)

printfn "Cyclist 1 average travel time: %A" posteriors5.[0].AverageTimeDist
printfn "Cyclist 1 traffic noise: %A"  posteriors5.[0].TrafficNoiseDist
printfn "Cyclist 1 average travel time: %A" posteriors5.[1].AverageTimeDist
printfn "Cyclist 2 traffic noise: %A" posteriors5.[1].TrafficNoiseDist
(*
Cyclist 1 average travel time: Gaussian(17.12, 2.741)
Cyclist 1 traffic noise: Gamma(6.712, 0.00536)[mean=0.03597]
Cyclist 1 average travel time: Gaussian(21.19, 2.722)
Cyclist 2 traffic noise: Gamma(6.4, 0.00577)[mean=0.03693]  
*)

// Make predictions based on the trained model
let cyclistsPrediction = TwoCyclistsPrediction()
//cyclistsPrediction.CreateModel()
cyclistsPrediction.SetModelData(posteriors5)

let posteriors5a = cyclistsPrediction.InferTomorrowsTime()

printfn "Cyclist1 tomorrow's travel time: %A" posteriors5a.[0]
printfn "Cyclist2 tomorrow's travel time: %A" posteriors5a.[1]
(*
Cyclist1 tomorrow's travel time: Gaussian(17.12, 35.4)
Cyclist2 tomorrow's travel time: Gaussian(21.19, 34.81)  
*)

let timeDifference = cyclistsPrediction.InferTimeDifference()
let cyclist1IsFaster = cyclistsPrediction.InferCyclist1IsFaster()

printfn "Time difference: %A" timeDifference
printfn "Probability that cyclist 1 is faster: %A" cyclist1IsFaster
(*
Time difference: Gaussian(-4.075, 70.22)
Probability that cyclist 1 is faster: Bernoulli(0.6866)  
*)

//--------------------------------------------------------------------
//Extras
//------

//Gamma------------------------------------
let dist1=Distributions.Gamma(shape=2.0,rate=1./0.5)
let samples1 = dist1.Samples() |> Seq.take 10000 |> Seq.toList
let histogram = Statistics.Histogram(samples1, 100)
Chart.Column (getValues histogram)
