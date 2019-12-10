using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MachineLearning.Model
{
    public class SpamInput
    {
        [LoadColumn(0)]  public string RawLabel { get; set; }
        [LoadColumn(1)]  public string Message { get; set; }
    }
    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")]  public bool IsSpam { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }
}
