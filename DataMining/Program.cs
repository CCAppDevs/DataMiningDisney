using Microsoft.ML;

namespace DataMining
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private IDataView trainingData;
        private string modelPath;
        private ITransformer trainedModel;

        // prediction engine (with input and output types)

        public Program()
        {
            // gather any variables and set them
            dataPath = Path.Combine(Environment.CurrentDirectory, "data\\DisneylandReviews.csv");
            modelPath = Path.Combine(Environment.CurrentDirectory, "models\\model.zip");

            // Create a context ()
            ctx = new MLContext();
            
            // read the input data into the system
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true);

            // build a data pipeline (transforming your data into something that works)

            // train your model (make it run the gauntlet)

            // Consume the model (make predictions)

            


            Console.ReadLine();
        }

        static void Main(string[] args)
        {
            new Program();
        }
    }
}
