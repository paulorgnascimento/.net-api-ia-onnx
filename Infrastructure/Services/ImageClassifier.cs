using ImageClassification.Domain.Entities;
using ImageClassification.Domain.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification.Infrastructure.Services
{
    public class ImageClassifier : IImageClassifier
    {
        private readonly InferenceSession _session;
        private readonly string[] _labels;

        public ImageClassifier()
        {
            var modelPath = Path.Combine(AppContext.BaseDirectory, "Infrastructure", "Models", "resnet50-v1-12.onnx");
            var labelsPath = Path.Combine(AppContext.BaseDirectory, "Infrastructure", "Models", "imagenet_classes.txt");


            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Modelo não encontrado: {modelPath}");

            if (!File.Exists(labelsPath))
                throw new FileNotFoundException($"Arquivo de classes não encontrado: {labelsPath}");

            _session = new InferenceSession(modelPath);
            _labels = File.ReadAllLines(labelsPath);
        }

        public PredictionResult ClassifyImage(ImageRequest request)
        {
            if (string.IsNullOrEmpty(request.Base64Image))
            {
                throw new ArgumentException("A imagem em Base64 é obrigatória.");
            }

            var base64Data = request.Base64Image.Contains(",") ? request.Base64Image.Split(',')[1] : request.Base64Image;
            var imageBytes = Convert.FromBase64String(base64Data);
            using var image = Image.Load<Rgb24>(imageBytes);

            var inputTensor = PreprocessImage(image);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", inputTensor)
            };
            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            var maxIndex = output.ToList().IndexOf(output.Max());
            var predictedLabel = _labels[maxIndex];
            var confidence = output[maxIndex];

            return new PredictionResult
            {
                Label = predictedLabel,
                Confidence = confidence
            };
        }

        private Tensor<float> PreprocessImage(Image<Rgb24> image)
        {
            const int TargetWidth = 224;
            const int TargetHeight = 224;

            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(TargetWidth, TargetHeight),
                Mode = ResizeMode.Crop
            }));

            var tensor = new DenseTensor<float>(new[] { 1, 3, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)
            {
                for (int x = 0; x < TargetWidth; x++)
                {
                    var pixel = image[x, y];
                    tensor[0, 0, y, x] = pixel.R / 255f;
                    tensor[0, 1, y, x] = pixel.G / 255f;
                    tensor[0, 2, y, x] = pixel.B / 255f;
                }
            }

            return tensor;
        }
    }
}