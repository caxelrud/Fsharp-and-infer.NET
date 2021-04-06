//rethink2_Ch4_1.fsx

#I @"C:\Users\inter\OneDrive\_myWork\Research2021\Infernet_2021\Packages"
#r "FSharp.Data.dll"
#r "Microsoft.ML.Probabilistic.Compiler.dll"
#r "Microsoft.ML.Probabilistic.dll"
#r "Microsoft.ML.Probabilistic.FSharp.dll"

#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"


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
open MathNet.Numerics
open MathNet.Numerics.Statistics


open FSharp.Data
open FSharp.Data.CsvExtensions


#r "System.Windows.Forms.DataVisualization.dll"
#nowarn "211"
#r "FSharp.Charting.dll"

open FSharp.Charting
module FsiAutoShow = 
    fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")


//------------------------------------------------------------------------------
//Height, weight dataset
//----------------------

//CSV Parser
//CSV columns: height, weight,age,male

let [<Literal>] data = __SOURCE_DIRECTORY__ + "\\Data\\Howell1_v2.csv"

let d1 = CsvFile.Load(data).Cache()

let f1=d1.Rows |> Seq.head
//val f1 : CsvRow = [|"151.765"; "47.8256065"; "63"; "1"|]
f1?height
//val it : string = "151.765"
for row in d1.Rows do
  printfn "R: (%A, %A, %A, %A)" row?height row?weight row?age row?male

let a1=[|for row in d1.Rows do
            [|(float row?height);(float row?weight);(float row?age);(float row?male)|] |]

let Height_18=
        Array.filter (fun (e:float array) -> e.[2] >=18.0) a1
        |> Array.map(fun (e:float array) -> e.[0])

let Height=
    Array.map(fun (e:float array) -> e.[0]) a1


let Weight_18=
    Array.filter (fun (e:float array) -> e.[2] >=18.0) a1
    |> Array.map(fun (e:float array) -> e.[1])

let Weight=
    Array.map(fun (e:float array) -> e.[1]) a1

//let Height=[| for row in d1.Rows -> (float row?height) |]
//let Weight=[| for row in d1.Rows -> (float row?weight) |]
//let Age=[| for row in d1.Rows -> (float row?age) |]
//let Male=[| for row in d1.Rows -> (int row?male) |]

//let Hmean=Statistics.Mean Height //138.2635963
//let Hstd=Statistics.StandardDeviation Height //27.60244764
let Hmean_18=Statistics.Mean Height_18 //154.5970926
let Hstd_18=Statistics.StandardDeviation Height_18 //7.742332137
let Wmean_18=Statistics.Mean Weight_18 //44.99048552
let Wstd_18=Statistics.StandardDeviation Weight_18 //6.456708107
let Wmean=Statistics.Mean Weight //44.99048552
let Wstd=Statistics.StandardDeviation Weight //6.456708107

let Weight_std=[|for r in Weight -> (r-Wmean)/Wstd|];;
let Weight_std2=[|for r in Weight_std -> r**2.0|];;

let L=Height.Length
let L_18=Height_18.Length

//Jump to the model to test avoiding the other models since that are repeated variables names 

(*
//MODEL 4.5 ----------------------------

d["weight_std"] = (d.weight - d.weight.mean()) / d.weight.std()
d["weight_std2"] = d.weight_std ** 2
with pm.Model() as m_4_5:
    a = pm.Normal("a", mu=178, sd=100)
    b1 = pm.Lognormal("b1", mu=0, sd=1)
    b2 = pm.Normal("b2", mu=0, sd=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = pm.Deterministic("mu", a + b1 * d.weight_std + b2 * d.weight_std2)
    height = pm.Normal("height", mu=mu, sd=sigma, observed=d.height)
*)
(*
Expected Posteriors:
	mean	sd	hdi_5.5%	hdi_94.5%
a	146.05	0.38	145.40	146.60
b2	-7.80	0.28	-8.26	-7.37
b1	21.74	0.29	21.30	22.20
sigma	5.80	0.18	5.52	6.08
*)

let R=Range(Height.GetLength(0))
let wmean= Variable.Constant<double>(Wmean).Named("wmean")
let wstd= Variable.Constant<double>(Wstd).Named("wstd")

let height= Variable.Observed<double>(Height,R).Named("height")
let weight= Variable.Observed<double>(Weight,R).Named("weight")
let weight_std= Variable.Observed<double>(Weight_std,R).Named("height_std")
let weight_std2= Variable.Observed<double>(Weight_std2,R).Named("weight_std2")

(*
let xbar=Variable.Sum(weight)/(float R.SizeAsInt)
*)

let a =  Variable.GaussianFromMeanAndVariance(150.0, 10.0**2.0).Named("a") 
let b1 =  Variable.GaussianFromMeanAndVariance(0.0, 5.0**2.0).Named("b1") 
Variable.ConstrainPositive(b1)
let b2 =  Variable.GaussianFromMeanAndVariance(0.0, 5.0**2.0).Named("b2") 

let prec =  Variable.GammaFromShapeAndScale(3.0, 3.0).Named("prec") 

let weight_stda= Variable.ArrayInit R
                    (fun r -> (weight.[r]-wmean)/wstd )

let weight_std2a= Variable.ArrayInit R
                    (fun r -> ((weight.[r]-wmean)/wstd)*((weight.[r]-wmean)/wstd) )

//Method 1
let mu= Variable.ArrayInit R
                    (fun r -> a + b1*weight_stda.[r] + b2*weight_std2a.[r])

//Method 2
//let mu= Variable.ArrayInit R
//                    (fun r -> a + b1*weight_std.[r] + b2*weight_std2.[r])

//Method 3
//let mu= Variable.ArrayInit R
//            (fun r -> a + b1*(weight.[r]-hmean)/hstd + b2*(weight.[r]-hmean)*(weight.[r]-hmean)/(hstd*hstd) )

mu.Name<-"mu"

Variable.AssignVariableArray height R
            (fun r -> Variable.GaussianFromMeanAndPrecision(mu.[r],prec))
            
let engine = InferenceEngine() //VariationalMessagePassing(),ExpectationPropagation(),VariationalMessagePassing() ,GibbsSampling() 
//engine.NumberOfIterations <- 100
//InferenceEngine.ShowFactorManager(false)

let aPost= engine.Infer<Gaussian>(a)
//val aPost : Gaussian = Gaussian(146.7, 0.06086)
let aPost_mean=aPost.GetMean()

let b1Post= engine.Infer<Gaussian>(b1)
//val b1Post : Gaussian = Gaussian(21.36, 0.0611)
let b1Post_mean=b1Post.GetMean()

let b2Post= engine.Infer<Gaussian>(b2)
//val b2Post : Gaussian = Gaussian(-8.425, 0.02972)
let b2Post_mean=b2Post.GetMean()

let precPost= engine.Infer<Gamma>(prec)
//val precPost : Gamma = Gamma(271.3, 0.0001121)[mean=0.03041]

let sigmaPost=sqrt(1.0/precPost.GetMean())
//val sigmaPost : float = 5.734238516


Chart.Combine(
        [Chart.Point [ for i in 0..L-1 -> (Weight.[i],Height.[i] )] 
         Chart.Point [ for i in 0..L-1 -> (Weight.[i], aPost_mean + b1Post_mean*Weight_std.[i] + b2Post_mean*Weight_std2.[i])  ]])


(*
//MODEL 4.3b ---------------------------
xbar = d2.weight.mean()
with pm.Model() as m4_3b:
a = pm.Normal("a", mu=178, sd=20)
b = pm.Normal("b", mu=0, sd=1)
sigma = pm.Uniform("sigma", 0, 50)
mu = a + np.exp(b) * (d2.weight - xbar)
height = pm.Normal("height", mu=mu, sd=sigma, observed=d2.height)
*)
(*
Expected Posteriors:(mean,sd,var,prec)
a:	(154.604,0.273,0.0745,13.41)	
b:	(0.904,0.041,0.001681,594.88)	
sigma:	(5.103,	0.201, 0.040401,24.75)	
*)


//let xbar=Statistics.Mean Weight_18 //44.99048552

let R=Range(Height_18.GetLength(0))
let height_18= Variable.Observed<double>(Height_18,R).Named("height_18")
let weight_18= Variable.Observed<double>(Weight_18,R).Named("weight_18")
let xbar=Variable.Sum(weight_18)/(float R.SizeAsInt)

let a =  Variable.GaussianFromMeanAndVariance(150.0, 20.0).Named("a") 
let b =  Variable.GaussianFromMeanAndVariance(1.0, 1.0).Named("b") 
Variable.ConstrainPositive(b)
let prec =  Variable.GammaFromShapeAndScale(3.0, 3.0).Named("prec") 
let mu= Variable.ArrayInit R
            (fun r -> a + b*(weight_18.[r]-xbar)) 
Variable.AssignVariableArray height_18 R
            (fun r -> Variable.GaussianFromMeanAndPrecision(mu.[r],prec))
            
let engine = InferenceEngine(ExpectationPropagation()) //VariationalMessagePassing() ,GibbsSampling() 
engine.ShowFactorGraph<-false

let aPost= engine.Infer<Gaussian>(a)
//val aPost : Gaussian = Gaussian(154.6, 0.07262)
aPost.GetVariance() //0.07262086934
let aPost_mean=aPost.GetMean()

let bPost= engine.Infer<Gaussian>(b)
//val bPost : Gaussian = Gaussian(0.9052, 0.001749)
let bPost_mean=bPost.GetMean()

let precPost= engine.Infer<Gamma>(prec)
//val precPost : Gamma = Gamma(177.1, 0.000222)[mean=0.03931]

let sigmaPost=sqrt(1.0/precPost.GetMean())
//val sigmaPost : float = 5.043389895

Chart.Combine(
    [Chart.Point( [ for i in 0..L_18-1 -> (Weight_18.[i],Height_18.[i] )]) 
     Chart.Point [ for i in 0..L_18-1 -> (Weight_18.[i],aPost_mean+ bPost_mean*(Weight_18.[i]-Wmean_18))] ]).WithYAxis(Min=125.0)



(*
//MODEL 4.1 ----------------------------
with pm.Model() as m4_1:
    mu = pm.Normal("mu", mu=178, sd=20)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    height = pm.Normal("height", mu=mu, sd=sigma, observed=d2.height)
*)
let mu_var=0.43**2.0 //0.1849
let mu_prec=1.0/mu_var //5.408328826
let sd_var=0.29**2.0 //0.0841
let sd_prec=1.0/sd_var //11.89060642
let prec_mu=1.0/(7.77**2.0) //0.01656372313
//let prec_sd= 1.0/0.29 //3.448275862
//let prec_var=1.0/0.0841 //141.3865211
//let prec_prec=1.0/prec_var //0.00707

(*
Posteriors:(mean,sd,var,prec)
mu:   (154.60,0.43,0.185,5.4)
sd:   (7.77  ,0.29,0.0841,11.89)
prec= (0.01656,3.44,11.89,0.0841)
*)
let mu =  Variable.GaussianFromMeanAndPrecision(178.0, 1.0/(20.0**2.0)).Named("mu") 
let prec =  Variable.GammaFromShapeAndScale(1.0, 1.0).Named("prec") 

let R= Range(L_18)
let height_18 = Variable.AssignVariableArray 
                    (Variable.Array<float>(R))  
                     R (fun d -> Variable.GaussianFromMeanAndPrecision(mu,prec))
height_18.ObservedValue <- Height_18    
height_18.Name<-"height_18"

let engine = InferenceEngine()
engine.ShowFactorGraph<-false

let muPost= engine.Infer<Gaussian>(mu)
//val muPost : Gaussian = Gaussian(154.6, 0.1702)
muPost.GetVariance()
//val it : float = 0.1702438385

let precPost= engine.Infer<Gamma>(prec)
//val precPost : Gamma = Gamma(176, 9.533e-05)[mean=0.01678]
precPost.GetVariance()
//val it : float = 1.599203773e-06
let sigmaPost=sqrt(1.0/precPost.GetMean())
//val sigmaPost : float = 7.720673996







