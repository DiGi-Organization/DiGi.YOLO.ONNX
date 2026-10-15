using DiGi.YOLO.Classes;
using DiGi.YOLO.ONNX.Classes;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace DiGi.YOLO.ONNX
{
    public static partial class Modify
    {
        /// <summary>
        /// Scores a directory of images against an exported ONNX model in this process and writes the detections to a bounding box result file.
        /// <para>This is the in-process counterpart of <see cref="YOLO.Modify.Predict(DiGi.YOLO.Classes.YOLOPredictionOptions?, CancellationToken)"/>, which runs the same detector through a CPython interpreter. Everything observable is deliberately the same: the images are taken in the order predict.py globs them, an image with nothing on it gets a line carrying only its name, a detection is written as name, label, corner, extents and confidence, and a stale result file is removed before anything is written so a failed run cannot be mistaken for this one. A source directory holding no images is answered without loading the model at all.</para>
        /// <para>Preprocessing goes through the same OpenCV that ultralytics calls through cv2 - the same JPEG decoder, the same bilinear resize - so the two paths differ only by the arithmetic of the graph itself rather than by what was fed into it.</para>
        /// <para>An image that will not decode is reported in <see cref="YOLOONNXPredictionResult.Messages"/> and given a line carrying only its name. Emgu 4.12's Imread throws instead of returning an empty Mat in that case, so the decode call is caught and both failure shapes are handled here rather than escaped; Ultralytics would end the run instead, and this keeps the result file aligned one-for-one with the source listing, which is what everything downstream of it assumes.</para>
        /// <para>Only the raw one-to-many head is decoded - four box values and one score per class for every anchor, suppressed here as ultralytics suppresses it. A graph exported as the end-to-end (NMS-free) head is refused before any image is read, because it is a different detector from the one the CPython path scores and its [batch, max_det, 6] layout would otherwise decode into nonsense without an error: the run fails with a message when the metadata carries end2end, or when the declared output is not three-dimensional or states no more anchors than channels.</para>
        /// <para>There is one known divergence, and it is reported rather than left silent. For a batch of equally shaped images ultralytics letterboxes onto the smallest canvas that is a multiple of the model stride, which for a non-square image is not a square; this path always pads onto a square. Every image the pipeline scores is 320 pixels square, so the two are the same transform and the divergence has never applied - a non-square source puts a note in <see cref="YOLOONNXPredictionResult.Messages"/> saying its detections may differ from the CPython path.</para>
        /// </summary>
        /// <param name="yOLOONNXPredictionOptions">The settings for the run.</param>
        /// <param name="cancellationToken">The token that cancels the run.</param>
        /// <returns>The result of the run, or <c>null</c> when the options are missing the model, the source directory or the output path.</returns>
        public static YOLOONNXPredictionResult? Predict(this YOLOONNXPredictionOptions? yOLOONNXPredictionOptions, CancellationToken cancellationToken = default)
        {
            if (yOLOONNXPredictionOptions == null)
            {
                return null;
            }

            string? modelPath = yOLOONNXPredictionOptions.ModelPath;
            string? sourceDirectory = yOLOONNXPredictionOptions.SourceDirectory;
            string? outputPath = yOLOONNXPredictionOptions.OutputPath;

            if (string.IsNullOrWhiteSpace(modelPath) || string.IsNullOrWhiteSpace(sourceDirectory) || string.IsNullOrWhiteSpace(outputPath))
            {
                return null;
            }

            modelPath = Python.Query.NormalizedPath(modelPath) ?? modelPath;
            sourceDirectory = Python.Query.NormalizedPath(sourceDirectory) ?? sourceDirectory;
            outputPath = Python.Query.NormalizedPath(outputPath) ?? outputPath;

            DateTimeOffset start = DateTimeOffset.Now;

            if (!Directory.Exists(sourceDirectory))
            {
                return new YOLOONNXPredictionResult(0, outputPath, null, [string.Format(CultureInfo.InvariantCulture, "Source directory does not exist: {0}", sourceDirectory)], start, DateTimeOffset.Now);
            }

            if (!File.Exists(modelPath))
            {
                return new YOLOONNXPredictionResult(0, outputPath, null, [string.Format(CultureInfo.InvariantCulture, "Model does not exist: {0}", modelPath)], start, DateTimeOffset.Now);
            }

            //The extensions predict.py globs for, in the order it globs them, so both paths list their images the same way
            List<string> paths_Image = [];
            foreach (string searchPattern in new string[] { "*.jpg", "*.jpeg", "*.png" })
            {
                paths_Image.AddRange(Directory.GetFiles(sourceDirectory, searchPattern));
            }

            if (paths_Image.Count == 0)
            {
                return new YOLOONNXPredictionResult(0, outputPath, [], null, start, DateTimeOffset.Now);
            }

            string? directory_Output = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(directory_Output) && !Directory.Exists(directory_Output))
            {
                Directory.CreateDirectory(directory_Output);
            }

            //A result file left by an earlier run would otherwise be read back as this run's answer
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            List<string> messages = [];
            List<string> values = [];

            bool cancelled = false;
            bool completed = false;
            bool reported_Padding = false;

            InferenceSession? inferenceSession;
            try
            {
                inferenceSession = new InferenceSession(modelPath);
            }
            catch (Exception exception)
            {
                return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, null, [string.Format(CultureInfo.InvariantCulture, "Model could not be loaded: {0}", exception.Message)], start, DateTimeOffset.Now);
            }

            try
            {
                //Ultralytics 8.4 exports a model that has a one-to-one head as that head when told nms=False - top-k boxes as [batch, max_det, 6], a different detector from
                //the one-to-many head the CPython path scores. Nothing further down would stop that layout: it decodes into nonsense and the run reports success. It is refused
                //by the flag ultralytics writes into the metadata and, for a graph carrying none, by its shape, since a raw head declares far more anchors than channels
                if (inferenceSession.ModelMetadata.CustomMetadataMap.TryGetValue("end2end", out string? end2End) && string.Equals(end2End?.Trim(), "True", StringComparison.OrdinalIgnoreCase))
                {
                    messages.Add("Model is an end-to-end (NMS-free) export answering [batch, max_det, 6], which this path cannot decode; re-export it with nms left at its default.");
                    return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, null, messages, start, DateTimeOffset.Now);
                }

                int[] dimensions_Output_Declared = inferenceSession.OutputMetadata[inferenceSession.OutputMetadata.Keys.First()].Dimensions;
                if (dimensions_Output_Declared.Length != 3 || (dimensions_Output_Declared[1] > 0 && dimensions_Output_Declared[2] > 0 && dimensions_Output_Declared[2] <= dimensions_Output_Declared[1]))
                {
                    messages.Add(string.Format(CultureInfo.InvariantCulture, "Model declares an output of [{0}], not the raw [batch, 4 + classes, anchors] layout this path decodes.", string.Join(", ", dimensions_Output_Declared)));
                    return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, null, messages, start, DateTimeOffset.Now);
                }

                string name_Input = inferenceSession.InputMetadata.Keys.First();
                int[] dimensions_Input = inferenceSession.InputMetadata[name_Input].Dimensions;

                //A model exported without dynamic axes states its batch and its canvas, and those beat anything the options ask for - feeding a fixed graph something else does not fail, it reshapes and answers nonsense
                int size = yOLOONNXPredictionOptions.InputSize;
                if (dimensions_Input.Length == 4 && dimensions_Input[2] > 0 && dimensions_Input[3] > 0 && dimensions_Input[2] == dimensions_Input[3])
                {
                    if (dimensions_Input[2] != size)
                    {
                        messages.Add(string.Format(CultureInfo.InvariantCulture, "Model declares a fixed input of {0} pixels; the requested {1} was ignored.", dimensions_Input[2], size));
                    }

                    size = dimensions_Input[2];
                }

                int count_Batch = yOLOONNXPredictionOptions.BatchSize < 1 ? 1 : yOLOONNXPredictionOptions.BatchSize;
                bool fixedBatch = dimensions_Input.Length == 4 && dimensions_Input[0] > 0;
                if (fixedBatch)
                {
                    count_Batch = dimensions_Input[0];
                }

                if (size < 1)
                {
                    messages.Add("Model input size could not be resolved.");
                    return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, null, messages, start, DateTimeOffset.Now);
                }

                //Counted in long, because the whole point of the ceiling is that this product overflows an int before it runs out of memory
                long count_Element_Image = 3L * size * size;
                long count_Element = count_Element_Image * count_Batch;

                if (count_Element > Constants.Count.MaximumInputElements)
                {
                    if (fixedBatch)
                    {
                        messages.Add(string.Format(CultureInfo.InvariantCulture, "Model declares a fixed batch of {0} at {1} pixels, which needs more input than one call is allowed to be handed.", count_Batch, size));
                        return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, null, messages, start, DateTimeOffset.Now);
                    }

                    int count_Batch_Clamped = (int)Math.Max(1, Constants.Count.MaximumInputElements / count_Element_Image);
                    messages.Add(string.Format(CultureInfo.InvariantCulture, "Batch size {0} needs more input than one call is allowed to be handed; {1} was used instead.", count_Batch, count_Batch_Clamped));
                    count_Batch = count_Batch_Clamped;
                }

                using (StreamWriter streamWriter = new(outputPath, false, new UTF8Encoding(false)))
                {
                    for (int i = 0; i < paths_Image.Count; i += count_Batch)
                    {
                        if (cancellationToken.IsCancellationRequested)
                        {
                            messages.Add("Prediction was cancelled.");
                            cancelled = true;
                            break;
                        }

                        int count_Chunk = Math.Min(count_Batch, paths_Image.Count - i);

                        //A fixed graph is fed a whole batch every time; the tail is padded by repeating its last real image, and the padding's answers are thrown away
                        int count_Fed = fixedBatch ? count_Batch : count_Chunk;

                        Mat[] mats = new Mat[count_Fed];
                        LetterBox?[] letterBoxes = new LetterBox?[count_Fed];

                        try
                        {
                            for (int j = 0; j < count_Fed; j++)
                            {
                                string path_Image = paths_Image[i + Math.Min(j, count_Chunk - 1)];

                                //Emgu 4.12 throws instead of returning an empty Mat when a file will not decode, so an undecodable image used to escape the per-image loop and abort the run. Both failure shapes land in the same branch.
                                Mat? mat = null;
                                try
                                {
                                    mat = CvInvoke.Imread(path_Image, ImreadModes.ColorBgr);
                                }
                                catch (ArgumentException)
                                {
                                    //A truncated or otherwise undecodable image is reported by Emgu this way
                                }
                                catch (CvException)
                                {
                                    //OpenCV's own decode failure, e.g. a 0-byte file
                                }

                                if (mat == null || mat.IsEmpty)
                                {
                                    mat?.Dispose();

                                    mats[j] = new Mat(size, size, DepthType.Cv8U, 3);
                                    mats[j].SetTo(new MCvScalar(114, 114, 114));
                                    letterBoxes[j] = null;

                                    if (j < count_Chunk)
                                    {
                                        messages.Add(string.Format(CultureInfo.InvariantCulture, "Image could not be decoded: {0}", path_Image));
                                    }

                                    continue;
                                }

                                LetterBox? letterBox = Create.LetterBox(mat.Width, mat.Height, size);
                                letterBoxes[j] = letterBox;

                                if (letterBox == null)
                                {
                                    mat.Dispose();

                                    mats[j] = new Mat(size, size, DepthType.Cv8U, 3);
                                    mats[j].SetTo(new MCvScalar(114, 114, 114));
                                    continue;
                                }

                                Mat mat_Resized = new();
                                CvInvoke.Resize(mat, mat_Resized, new Size(letterBox.Width, letterBox.Height), 0, 0, Inter.Linear);
                                mat.Dispose();

                                if (letterBox.Width == size && letterBox.Height == size)
                                {
                                    mats[j] = mat_Resized;
                                    continue;
                                }

                                //Padding means the source was not square, and that is the one case where this path knowingly parts
                                //company with the CPython one. Ultralytics letterboxes a batch of equally shaped images onto the
                                //smallest canvas that is still a multiple of the stride, which for a non-square image is not a
                                //square; this pads to a full square. The detections then differ slightly, and would do so silently,
                                //so the run says it happened. Every image this pipeline scores is 320 pixels square, which is why
                                //this has never fired - and exactly why it would be missed if it ever did.
                                if (!reported_Padding && j < count_Chunk)
                                {
                                    reported_Padding = true;
                                    messages.Add(string.Format(CultureInfo.InvariantCulture, "Image is not square ({0} by {1}), so it was padded onto a square canvas: {2}. Ultralytics would have used a smaller canvas for a batch of this shape, so detections on images like this may differ from the CPython path.", letterBox.SourceWidth, letterBox.SourceHeight, path_Image));
                                }

                                Mat mat_Padded = new();
                                CvInvoke.CopyMakeBorder(mat_Resized, mat_Padded, letterBox.OffsetY, size - letterBox.Height - letterBox.OffsetY, letterBox.OffsetX, size - letterBox.Width - letterBox.OffsetX, BorderType.Constant, new MCvScalar(114, 114, 114));
                                mat_Resized.Dispose();

                                mats[j] = mat_Padded;
                            }

                            float[] values_Input = new float[count_Fed * 3 * size * size];

                            using (Mat mat_Blob = DnnInvoke.BlobFromImages(mats, 1d / 255d, new Size(size, size), new MCvScalar(0, 0, 0), true, false, DepthType.Cv32F))
                            {
                                Marshal.Copy(mat_Blob.DataPointer, values_Input, 0, values_Input.Length);
                            }

                            DenseTensor<float> denseTensor_Input = new(new Memory<float>(values_Input), [count_Fed, 3, size, size]);

                            List<NamedOnnxValue> namedOnnxValues = [NamedOnnxValue.CreateFromTensor(name_Input, denseTensor_Input)];

                            float[] values_Output;
                            int[] dimensions_Output;

                            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> disposableNamedOnnxValues = inferenceSession.Run(namedOnnxValues))
                            {
                                Tensor<float> tensor_Output = disposableNamedOnnxValues[0].AsTensor<float>();

                                dimensions_Output = tensor_Output.Dimensions.ToArray();

                                int count_Value = 1;
                                foreach (int dimension in dimensions_Output)
                                {
                                    count_Value *= dimension;
                                }

                                DenseTensor<float> denseTensor_Output = tensor_Output as DenseTensor<float> ?? tensor_Output.ToDenseTensor();
                                values_Output = denseTensor_Output.Buffer.Span.Slice(0, count_Value).ToArray();
                            }

                            for (int j = 0; j < count_Chunk; j++)
                            {
                                string name = Path.GetFileNameWithoutExtension(paths_Image[i + j]);

                                List<BoundingBoxResult>? boundingBoxResults = letterBoxes[j] == null ? null : Query.Detections(values_Output, dimensions_Output, j, name, letterBoxes[j], yOLOONNXPredictionOptions);

                                if (boundingBoxResults == null || boundingBoxResults.Count == 0)
                                {
                                    streamWriter.WriteLine(name);
                                    values.Add(name);
                                    continue;
                                }

                                foreach (BoundingBoxResult boundingBoxResult in boundingBoxResults)
                                {
                                    string value = boundingBoxResult.ToString();
                                    streamWriter.WriteLine(value);
                                    values.Add(value);
                                }
                            }
                        }
                        finally
                        {
                            foreach (Mat mat in mats)
                            {
                                mat?.Dispose();
                            }
                        }

                        streamWriter.Flush();
                    }
                }

                completed = !cancelled;
            }
            catch (Exception exception)
            {
                messages.Add(string.Format(CultureInfo.InvariantCulture, "Prediction failed: {0}", exception.Message));
            }
            finally
            {
                inferenceSession.Dispose();
            }

            //A run that did not reach the end leaves a half-written file behind, and that file would be read back as this run's answer
            if (!completed)
            {
                if (File.Exists(outputPath))
                {
                    File.Delete(outputPath);
                }

                return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, null, messages, start, DateTimeOffset.Now);
            }

            return new YOLOONNXPredictionResult(paths_Image.Count, outputPath, values, messages.Count == 0 ? null : messages, start, DateTimeOffset.Now);
        }
    }
}
