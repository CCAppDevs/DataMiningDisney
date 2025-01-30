using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System.Data;

namespace DataMining
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private string testDataPath;
        private IDataView trainingData;
        private IDataView testData;
        private string modelPath;
        private ITransformer trainedModel;
        private EstimatorChain<KeyToValueMappingTransformer> pipeline;
        private bool IsRunning = true;

        // prediction engine (with input and output types)

        public Program()
        {
            // gather any variables and set them
            dataPath = Path.Combine(Environment.CurrentDirectory, "data\\DisneylandReviews.csv");
            testDataPath = Path.Combine(Environment.CurrentDirectory, "data\\testData.csv");
            modelPath = Path.Combine(Environment.CurrentDirectory, "data", "model.zip");

            // Create a context ()
            ctx = new MLContext();
            
            // read the input data into the system
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true, separatorChar: ',');

            CreatePipeline();

            TrainAndSaveModel();
        
            while (IsRunning)
            {
                Console.Clear();

                Console.WriteLine("-------------------");
                Console.WriteLine("Main Menu");
                Console.WriteLine("-------------------");
                Console.WriteLine("1. Retrain Model");
                Console.WriteLine("2. Evalutate Model");
                Console.WriteLine("3. Make a Prediction");
                Console.WriteLine("0. Quit");

                int choice = -1;

                Console.Write("What would you like to do? ");
                choice = Int32.Parse(Console.ReadLine());

                switch (choice)
                {
                    case 0:
                        IsRunning = false;
                        break;
                    case 1:
                        TrainAndSaveModel();
                        Console.WriteLine("Model has been retrained and saved. Press enter to continue.");
                        Console.ReadLine();
                        break;
                    case 2:
                        Evaluate();
                        Console.WriteLine("Press enter to continue.");
                        Console.ReadLine();
                        break;
                    case 3:
                        MakePrediction();
                        Console.WriteLine("Press enter key to continue.");
                        Console.ReadLine();
                        break;
                    default:
                        Console.WriteLine("Invalid Choice: Press enter to try again.");
                        Console.ReadLine();
                        break;
                }
            }

        }

        private void MakePrediction()
        {
            Console.Write("What is the text of the review?");
            string reviewText = Console.ReadLine();

            // capture text/data to be predicted
            var review = new DisneylandReview {
                ReviewText = reviewText
            };

            var prediction = Predict(review);
            PrintPrediction(prediction, review);

            // ask if its accurate
            // if it is, add it to our data set tagged with the correct values.
            // if not, correct the values, add to the data set, and retrain.
        }

        private void Evaluate()
        {
            var schema = trainingData.Schema;

            testData = ctx.Data.LoadFromTextFile<DisneylandReview>(testDataPath, hasHeader: true, separatorChar: ',');

            var testMetrics = ctx.MulticlassClassification.Evaluate(trainedModel.Transform(testData));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private void TrainAndSaveModel()
        {
            // train your model (make it run the gauntlet)
            trainedModel = pipeline.Fit(trainingData);

            SaveModelAsFile(ctx, trainingData.Schema, trainedModel);
        }

        private void CreatePipeline()
        {
            // build a data pipeline (transforming your data into something that works)
            pipeline = ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Rating", outputColumnName: "Label")
                .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "ReviewText", outputColumnName: "FeaturizedReviewText"))
                .Append(ctx.Transforms.Concatenate("Features", "FeaturizedReviewText"))
                .AppendCacheCheckpoint(ctx)
                .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }

        void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            Console.WriteLine("Writing model to disk");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
            Console.WriteLine("Finished Writing Model");
        }

        public void PrintPrediction(DisneylandPrediction prediction, DisneylandReview review)
        {
            Console.WriteLine($"" +
                $"Review Text: {review.ReviewText}\n" +
                $"Predicted Rating: {prediction.Prediction}\n" +
                $"Actual Rating: {review.Rating}\n");

            for (int i = 0; i < prediction.Score.Length; i++)
            {
                Console.WriteLine($"Score {i}: {prediction.Score[i]}");
            }

        }

        public DisneylandPrediction? Predict(DisneylandReview review)
        {
            var predictionEngine = ctx.Model.CreatePredictionEngine<DisneylandReview, DisneylandPrediction>(trainedModel);
            var prediction = predictionEngine.Predict(review);

            return prediction;
        }

        static void Main(string[] args)
        {
            new Program();
        }
    }
}
