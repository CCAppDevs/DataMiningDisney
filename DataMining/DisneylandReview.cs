using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining
{
    public class DisneylandReview
    {
        [LoadColumn(0)]
        public int ReviewId { get; set; }
        [LoadColumn(1)]
        public int Rating { get; set; }
        [LoadColumn(2)]
        public DateOnly YearMonth { get; set; }
        [LoadColumn(3)]
        public string ReviewerLocation { get; set; }
        [LoadColumn(4)]
        public string ReviewText { get; set; }
        [LoadColumn(5)]
        public string Branch { get; set; }
    }
}
