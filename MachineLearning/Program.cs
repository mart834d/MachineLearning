using System;
using System.IO;
using System.Linq;
using MachineLearning.Model;
using Microsoft.ML;
using static MachineLearning.Model.MachineLearning;

namespace MachineLearning
{
    public class Program
    {

        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "spam.csv");

        static void Main(string[] args)
        {
            var context = new MLContext();
            // data fra spam.csv(træningsæt) bliver læst in i memorien
            var data = context.Data.LoadFromTextFile<SpamInput>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',');
            // use 80% for training and 20% for testing
            var partitions = context.Data.TrainTestSplit(
                data,
                testFraction: 0.2);

            // Først så laver den spam, eller ham om en til en bool'
            // Som vores model kan arbejde med.
            // featurized text hjælper med hvordan model skal forstå data
            // til sidst bruger jeg en metode, der hjælper på at lave mere præsise forudsigelser
            var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>(
        mapAction: (input, output) => { output.Label = input.RawLabel == "spam" ? true : false; },
     contractName: "MyLambda")

            .Append(context.Transforms.Text.FeaturizeText(
         outputColumnName: "Features",
         inputColumnName: nameof(SpamInput.Message)))

            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());



            // her træner jeg min model.
            var model = pipeline.Fit(partitions.TrainSet);


            // Evaluerer min model
            Console.WriteLine("Evalulerer modellen");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = context.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score");


            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();


            Console.WriteLine("Predicter udfald af beskeder");
            var predictionEngine = context.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);

            var messages = new SpamInput[] {
                new SpamInput() { Message = "Hello, win free iPad" },
                new SpamInput() { Message = "Im home in few" },
                new SpamInput() { Message = "Hundreds of medications all 80% off or more!"},
                new SpamInput() { Message = "CONGRATS U WON LOTERY CLAIM UR 1 MILIONN DOLARS PRIZE" },};

            var myPredictions = from m in messages
                                select (Message: m.Message, Prediction: predictionEngine.Predict(m));

            foreach (var p in myPredictions)
                Console.WriteLine($"  [{p.Prediction.Probability:P2}] {p.Message}");

            Console.WriteLine("Skriv noget spam eller ham :");

            var userInput = Console.ReadLine();
            userMessage(userInput, predictionEngine);
        }
        static void userMessage(string message, PredictionEngine<SpamInput, SpamPrediction> predictionEngine)
        {
            var messages = new SpamInput[] {
                new SpamInput() { Message = message }};

            var myPredictions = from m in messages
                                select (Message: m.Message, Prediction: predictionEngine.Predict(m));

            foreach (var p in myPredictions)
                Console.WriteLine($"  [{p.Prediction.Probability:P2}] {p.Message}");

            Console.WriteLine("Skriv noget spam eller ham: ");

            var userInput = Console.ReadLine();
            userMessage(userInput, predictionEngine);
        }
    }
}
