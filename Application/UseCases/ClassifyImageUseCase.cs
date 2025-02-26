using ImageClassification.Domain.Entities;
using ImageClassification.Domain.Interfaces;

namespace ImageClassification.Application.UseCases
{
    public class ClassifyImageUseCase
    {
        private readonly IImageClassifier _imageClassifier;

        public ClassifyImageUseCase(IImageClassifier imageClassifier)
        {
            _imageClassifier = imageClassifier;
        }

        public PredictionResult Execute(ImageRequest request)
        {
            return _imageClassifier.ClassifyImage(request);
        }
    }
}