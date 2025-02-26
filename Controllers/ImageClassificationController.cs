using ImageClassification.Application.UseCases;
using ImageClassification.Domain.Entities;
using Microsoft.AspNetCore.Mvc;
using System;

namespace ImageClassification.Presentation.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ImageClassificationController : ControllerBase
    {
        private readonly ClassifyImageUseCase _classifyImageUseCase;

        public ImageClassificationController(ClassifyImageUseCase classifyImageUseCase)
        {
            _classifyImageUseCase = classifyImageUseCase;
        }

        [HttpPost("classify")]
        public IActionResult ClassifyImage([FromBody] ImageRequest request)
        {
            try
            {
                var result = _classifyImageUseCase.Execute(request);
                return Ok(result);
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Erro ao processar a imagem: {ex.Message}");
            }
        }
    }
}