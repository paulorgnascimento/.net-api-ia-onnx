using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ImageClassificationController : ControllerBase
    {
        private readonly InferenceSession _session;
        private readonly string[] _labels;

        public ImageClassificationController()
        {
            // Caminho correto dentro do contêiner
            var modelPath = Path.Combine(AppContext.BaseDirectory, "Controllers", "resnet50-v1-12.onnx");
            var labelsPath = Path.Combine(AppContext.BaseDirectory, "Controllers", "imagenet_classes.txt");

            if (!System.IO.File.Exists(modelPath))
                throw new FileNotFoundException($"Modelo não encontrado: {modelPath}");

            if (!System.IO.File.Exists(labelsPath))
                throw new FileNotFoundException($"Arquivo de classes não encontrado: {labelsPath}");

            _session = new InferenceSession(modelPath);
            _labels = System.IO.File.ReadAllLines(labelsPath);
        }

        [HttpPost("classify")]
        public IActionResult ClassifyImage([FromBody] ImageRequest request)
        {
            if (string.IsNullOrEmpty(request.Base64Image))
            {
                return BadRequest("A imagem em Base64 é obrigatória.");
            }

            try
            {
                // Remover o prefixo "data:image/jpeg;base64," se estiver presente
                var base64Data = request.Base64Image;
                if (base64Data.Contains(","))
                {
                    base64Data = base64Data.Substring(base64Data.IndexOf(",") + 1);
                }

                // Converter Base64 para imagem
                var imageBytes = Convert.FromBase64String(base64Data);
                using var image = Image.Load<Rgb24>(imageBytes);

                // Pré-processar a imagem
                var inputTensor = PreprocessImage(image);

                // Fazer a inferência
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("data", inputTensor)
                };
                using var results = _session.Run(inputs);
                var output = results.First().AsEnumerable<float>().ToArray();

                // Obter a previsão
                var maxIndex = output.ToList().IndexOf(output.Max());
                var predictedLabel = _labels[maxIndex];
                var confidence = output[maxIndex];

                return Ok(new PredictionResult
                {
                    Label = predictedLabel,
                    Confidence = confidence
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Erro ao processar a imagem: {ex.Message}");
            }
        }

        private Tensor<float> PreprocessImage(Image<Rgb24> image)
        {
            const int TargetWidth = 224;
            const int TargetHeight = 224;

            // Redimensionar a imagem
            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(TargetWidth, TargetHeight),
                Mode = ResizeMode.Crop
            }));

            // Normalizar para o intervalo [0, 1] e converter para tensor
            var tensor = new DenseTensor<float>(new[] { 1, 3, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)
            {
                for (int x = 0; x < TargetWidth; x++)
                {
                    var pixel = image[x, y];

                    tensor[0, 0, y, x] = pixel.R / 255f; // Canal R
                    tensor[0, 1, y, x] = pixel.G / 255f; // Canal G
                    tensor[0, 2, y, x] = pixel.B / 255f; // Canal B
                }
            }

            return tensor;
        }
    }
}
