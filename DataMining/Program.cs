using Microsoft.ML;

namespace DataMining
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private string testDataPath;
        private IDataView trainingData;
        private string modelPath;
        private ITransformer trainedModel;

        // prediction engine (with input and output types)

        public Program()
        {
            // gather any variables and set them
            dataPath = Path.Combine(Environment.CurrentDirectory, "data\\DisneylandReviews.csv");
            testDataPath = Path.Combine(Environment.CurrentDirectory, "data\\testData.csv");
            modelPath = Path.Combine(Environment.CurrentDirectory, "models\\model.zip");

            // Create a context ()
            ctx = new MLContext();
            
            // read the input data into the system
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true, separatorChar: ',');

            // build a data pipeline (transforming your data into something that works)
            var pipeline = ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Rating", outputColumnName: "Label")
                .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "ReviewText", outputColumnName: "FeaturizedReviewText"))
                .Append(ctx.Transforms.Concatenate("Features", "FeaturizedReviewText"))
                .AppendCacheCheckpoint(ctx)
                .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            // build and train your model (make it run the gauntlet)

            trainedModel = pipeline.Fit(trainingData);
            Evaluate(trainingData.Schema);
            // Consume the model (make predictions)

            // capture some text
            var sampleStatement = new DisneylandReview
            {
                ReviewId = 2718239,
                Rating = 1,
                YearMonth = "missing",
                ReviewerLocation = "United Kingdom",
                ReviewText = "Tickets were cheaper than other Disneys. Fast Pass was the bomb as we pre bought. The best rides were Mansion and the Grizzly Man Coaster. Space Mountain is not as good as Florida, California and even Paris. Ironman is okay. Pooh bear is good. This park is way smaller than the others which keeps the day simple. You only need one day here.",
                Branch = "Disneyland_Paris"
            };


            var prediction = Predict(sampleStatement);

            PrintPrediction(prediction, sampleStatement);

            Console.ReadLine();
        }

        public void PrintPrediction(DisneylandPrediction prediction, DisneylandReview review)
        {
            Console.WriteLine($"" +
                $"Review Text: {review.ReviewText}\n" +
                $"Predicted Rating: {prediction.Prediction}\n" +
                $"Actual Rating: {review.Rating}\n");

            for (int i = 0; i < prediction.Score.Length; i++)
            {
                Console.WriteLine($"Score {i + 1}: {prediction.Score[i]}");
            }
        }

        public void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = ctx.Data.LoadFromTextFile<DisneylandReview>(testDataPath, hasHeader: true, separatorChar: ',');
            var testMetrics = ctx.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
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
