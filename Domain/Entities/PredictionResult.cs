namespace ImageClassification.Domain.Entities
{
    public class PredictionResult
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
    }
}
