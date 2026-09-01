using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using CarClassificationApp.Models;
using CarClassificationApp.Services;

namespace CarClassificationApp.Controllers;

public class HomeController : Controller
{
    private readonly InferenceService _inferenceService;

    public HomeController(InferenceService inferenceService)
    {
        _inferenceService = inferenceService;
    }

    public IActionResult Index()
    {
        return View();
    }

    [HttpPost]
    public IActionResult Classify(IFormFile image)
    {
        if (image == null || image.Length == 0)
        {
            return BadRequest("Lütfen bir resim yükleyin.");
        }

        using var stream = image.OpenReadStream();
        var probabilities = _inferenceService.Predict(stream);

        var results = _inferenceService.ClassNames
            .Select((name, index) => new { ClassName = name, Probability = probabilities[index] })
            .OrderByDescending(x => x.Probability)
            .ToList();

        return Json(new
        {
            predictedClass = results.First().ClassName,
            probabilities = results
        });
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
