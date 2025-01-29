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
            modelPath = Path.Combine(Environment.CurrentDirectory, "models", "model");

            // Create a context ()
            ctx = new MLContext(200);
            
            // read the input data into the system
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true, separatorChar: ',');

            // build a data pipeline (transforming your data into something that works)
            var pipeline = ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Rating", outputColumnName: "Label")
                .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "ReviewText", outputColumnName: "FeaturizedReviewText"))
                .Append(ctx.Transforms.Concatenate("Features", "FeaturizedReviewText"))
                .AppendCacheCheckpoint(ctx)
                .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            // train your model (make it run the gauntlet)
            trainedModel = pipeline.Fit(trainingData);

            SaveModelAsFile(ctx, trainingData.Schema, trainedModel);

            // Consume the model (make predictions)

            // capture some text
            //var sampleStatement = new DisneylandReview
            //{
            //    ReviewId = 2842749,
            //    Rating = "1",
            //    YearMonth = "missing",
            //    ReviewerLocation = "United Kingdom",
            //    ReviewText = "this place is the best ever",
            //    Branch = "Disneyland_Paris"
            //};


            //var prediction = Predict(sampleStatement);

            //PrintPrediction(prediction, sampleStatement);

            //Console.ReadLine();
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
