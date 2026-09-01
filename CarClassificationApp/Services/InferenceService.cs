using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CarClassificationApp.Services
{
    public class InferenceService
    {
        private readonly InferenceSession _session;
        // Adjust the class order to match the alphabetical order of ImageFolder or as provided
        // Based on the user's kaggle dataset: "ACIK TEKERLEKLI", "HATCHBACK", "MICRO", "PICK UP", "SEDAN", "STATION WAGON", "SUV", "VAN" 
        // We will stick to the PDF's exact spelling or close variations
        public readonly string[] ClassNames = new string[] 
        {
            "AÇIK TEKERLEKLİ (F1 ARAÇLARI)",
            "HATCHBACK",
            "MİCRO",
            "PICK UP",
            "SEDAN",
            "STATION WAGON",
            "SUV",
            "VAN"
        };

        public InferenceService(string modelPath)
        {
            _session = new InferenceSession(modelPath);
        }

        public float[] Predict(Stream imageStream)
        {
            using var image = Image.Load<Rgb24>(imageStream);
            // Kaggle'daki PyTorch modeliyle BİREBİR aynı preprocessing:
            // 1. En boy oranını yoksayarak 256x256'ya sündür (Stretch)
            // 2. Ortadan 224x224 boyutunda kes (Center Crop)
            image.Mutate(x => 
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(256, 256),
                    Mode = ResizeMode.Stretch
                });
                
                // (256 - 224) / 2 = 16 (X ve Y ekseninden 16 piksel boşluk bırakıp ortayı alırız)
                x.Crop(new Rectangle(16, 16, 224, 224));
            });

            // Preprocess and create tensor
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / std[0];
                        input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / std[1];
                        input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / std[2];
                    }
                }
            });

            // Run inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            
            var output = results.First().AsEnumerable<float>().ToArray();
            
            // Apply Softmax
            return Softmax(output);
        }

        private float[] Softmax(float[] z)
        {
            var exp = z.Select(x => Math.Exp(x)).ToArray();
            var sumExp = exp.Sum();
            return exp.Select(x => (float)(x / sumExp)).ToArray();
        }
    }
}
