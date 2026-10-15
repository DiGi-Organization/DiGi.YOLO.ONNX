using DiGi.YOLO.ONNX.Classes;
using System.IO;

namespace DiGi.YOLO.ONNX
{
    public static partial class Create
    {
        /// <summary>
        /// Builds the options for one in-process ONNX prediction run, normalizing the paths and then checking that the combination can actually make a run.
        /// <para>The <see cref="Classes.YOLOONNXPredictionOptions"/> constructors only assign, so this is where the work belongs. The thresholds keep their defaults, which are the ones ultralytics applies.</para>
        /// </summary>
        /// <param name="modelPath">The path of the exported ONNX model to score with.</param>
        /// <param name="sourceDirectory">The directory holding the images to score.</param>
        /// <param name="outputPath">The path of the bounding box result file to write.</param>
        /// <param name="confidence">The confidence threshold a detection has to reach to be reported.</param>
        /// <returns>The options, or <c>null</c> when a required path is missing or the confidence is not a value between zero and one.</returns>
        public static YOLOONNXPredictionOptions? YOLOONNXPredictionOptions(string? modelPath, string? sourceDirectory, string? outputPath, double confidence = 0.1)
        {
            return YOLOONNXPredictionOptions(modelPath, sourceDirectory, outputPath, confidence, 0.7, 8);
        }

        /// <summary>
        /// Builds the options for one in-process ONNX prediction run with custom thresholds, normalizing the paths and then checking that the combination can actually make a run.
        /// <para>The <see cref="Classes.YOLOONNXPredictionOptions"/> constructors only assign, so this is where the work belongs.</para>
        /// <para>Moving either threshold away from ultralytics' default makes this path answer differently from the CPython one on purpose. That is a legitimate thing to want and an illegitimate thing to do by accident, which is why they are parameters here rather than constants.</para>
        /// </summary>
        /// <param name="modelPath">The path of the exported ONNX model to score with.</param>
        /// <param name="sourceDirectory">The directory holding the images to score.</param>
        /// <param name="outputPath">The path of the bounding box result file to write.</param>
        /// <param name="confidence">The confidence threshold a detection has to reach to be reported.</param>
        /// <param name="iou">The overlap above which the weaker of two detections of the same class is discarded.</param>
        /// <param name="batchSize">The number of images handed to the session in a single inference call.</param>
        /// <returns>The options, or <c>null</c> when a required path is missing, the confidence or the overlap is not a value between zero and one, or the batch size is less than one.</returns>
        public static YOLOONNXPredictionOptions? YOLOONNXPredictionOptions(string? modelPath, string? sourceDirectory, string? outputPath, double confidence, double iou, int batchSize)
        {
            string? modelPath_Resolved = Python.Query.NormalizedPath(modelPath);
            string? sourceDirectory_Resolved = Python.Query.NormalizedPath(sourceDirectory);
            string? outputPath_Resolved = Python.Query.NormalizedPath(outputPath);

            if (string.IsNullOrWhiteSpace(modelPath_Resolved) || !File.Exists(modelPath_Resolved))
            {
                return null;
            }

            if (string.IsNullOrWhiteSpace(sourceDirectory_Resolved) || !Directory.Exists(sourceDirectory_Resolved))
            {
                return null;
            }

            if (string.IsNullOrWhiteSpace(outputPath_Resolved))
            {
                return null;
            }

            if (double.IsNaN(confidence) || confidence < 0 || confidence > 1)
            {
                return null;
            }

            if (double.IsNaN(iou) || iou < 0 || iou > 1 || batchSize < 1)
            {
                return null;
            }

            return new YOLOONNXPredictionOptions()
            {
                BatchSize = batchSize,
                Confidence = confidence,
                IoU = iou,
                ModelPath = modelPath_Resolved,
                OutputPath = outputPath_Resolved,
                SourceDirectory = sourceDirectory_Resolved
            };
        }
    }
}
