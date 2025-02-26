using ImageClassification.Domain.Entities;

namespace ImageClassification.Domain.Interfaces
{
    public interface IImageClassifier
    {
        PredictionResult ClassifyImage(ImageRequest request);
    }
}