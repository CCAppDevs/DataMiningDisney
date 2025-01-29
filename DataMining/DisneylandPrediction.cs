using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining
{
    public class DisneylandPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }

        public float[] Probability { get; set; }

        public float[] Score { get; set; }
    }
}
