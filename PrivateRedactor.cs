using System.Diagnostics;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.DotNet;

namespace PrivateRedactor
{
    public class PrivateRedactor : IPrivateRedactor
    {
        private readonly ILogger _logger;
        private static InferenceSession? _session;
        private static Tokenizer? _tokenizer;
        private static Dictionary<int, string> Id2Label;

        private static int currentThreads = 0;

        public PrivateRedactor(ILogger logger)
        {
            _logger = logger;
        }

        public RedactionResult Redact(
            string Text,
            string DetectableEntitiesOverride = "",
            int Threads = 1
        )
        {

            if (Threads < 1) throw new ArgumentException("Threads must be greater or equal than 1");

            var result = new RedactionResult();
            if (string.IsNullOrEmpty(Text))
            {
                result.RedactedText = Text;
                return result;
            }

            if (_tokenizer is null)
            {
                string configJsonPath = Path.Combine(AppContext.BaseDirectory, "model", "config.json");
                Id2Label = ModelConfigLoader.LoadId2Label(configJsonPath);

                string tokenizerJsonPath = Path.Combine(AppContext.BaseDirectory, "model", "tokenizer.json");
                _tokenizer = new Tokenizer(tokenizerJsonPath);
            }

            if (_session is null || Threads != currentThreads)
            {
                InitModel(Threads);
                currentThreads = Threads;
            }

            Stopwatch sw = Stopwatch.StartNew();

            // 1. Get token IDs
            uint[] encodedTokens = _tokenizer.Encode(Text);

            const int MAX_TOKEN_LIMIT = 511;
            if (encodedTokens.Length > MAX_TOKEN_LIMIT)
            {
                throw new ArgumentException(
                    "The input text exceeds the model's maximum context limit of 512 tokens. " +
                    "Please chunk your text before processing.");
            }

            long[] inputIds = encodedTokens.Select(t => (long)t).ToArray();
            long[] attentionMask = Enumerable.Repeat(1L, inputIds.Length).ToArray();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", CreateTensor(inputIds)),
                NamedOnnxValue.CreateFromTensor("attention_mask", CreateTensor(attentionMask))
            };

            // 2. Run model inference
            using var results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            int numClasses = outputTensor.Dimensions[2];
            float[] flatLogits = outputTensor.ToArray();

            // 3. Reconstruct structural character offsets manually
            List<(int Start, int End)> tokenOffsets = GenerateOffsetsFromTokens(Text, encodedTokens);

            // 4. Collect Raw PII Entity hits matching character boundaries
            var rawEntities = new List<DetectedEntity>();

            string[] detectableEntities = DetectableEntitiesOverride.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < encodedTokens.Length; i++)
            {
                if (i >= tokenOffsets.Count) break;
                var offset = tokenOffsets[i];
                if (offset.Start == offset.End) continue; // Skip padding or spaces that have zero semantic length

                int maxClassId = 0;
                float maxScore = float.MinValue;
                int tokenOffset = i * numClasses;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = flatLogits[tokenOffset + c];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = c;
                    }
                }

                string labelName = Id2Label.ContainsKey(maxClassId) ? Id2Label[maxClassId] : "O";

                if (labelName != "O")
                {
                    string cleanTag = labelName.Replace("B-", "").Replace("I-", "").ToUpper();

                    if(detectableEntities.Length > 0 && !detectableEntities.Contains(cleanTag, StringComparer.OrdinalIgnoreCase))
                    {
                        continue; // Skip entities that are not in the user-specified detectable list
                    }

                    rawEntities.Add(new DetectedEntity
                    {
                        Start = offset.Start,
                        End = offset.End,
                        Label = cleanTag,
                        Text = Text.Substring(offset.Start, offset.End - offset.Start)
                    });
                }
            }

            // 5. Merge adjacent tokens with identical labels
            List<DetectedEntity> aggregatedEntities = AggregateEntities(rawEntities, Text);
            result.Entities = aggregatedEntities;

            // 6. Apply Redaction via Reverse Pass
            StringBuilder sb = new StringBuilder(Text);
            var reverseEntities = aggregatedEntities.OrderByDescending(e => e.Start).ToList();

            foreach (var entity in reverseEntities)
            {
                int length = entity.End - entity.Start;
                string replacementTag = $"[{entity.Label}]";

                sb.Remove(entity.Start, length);
                sb.Insert(entity.Start, replacementTag);
            }

            sw.Stop();

            result.Duration = sw.ElapsedMilliseconds;
            result.RedactedText = sb.ToString();
            return result;
        }

        private static List<DetectedEntity> AggregateEntities(List<DetectedEntity> rawEntities, string originalText)
        {
            if (!rawEntities.Any()) return new List<DetectedEntity>();

            var sorted = rawEntities.OrderBy(e => e.Start).ToList();
            var merged = new List<DetectedEntity>();
            var current = sorted[0];

            for (int i = 1; i < sorted.Count; i++)
            {
                var next = sorted[i];

                bool isSameLabel = next.Label == current.Label;

                // FIX: Increase adjacency threshold to 6 characters. 
                // This forces the aggregator to swallow short fragments (like "111") 
                // that sit directly inside a single PII entity run.
                bool isAdjacent = next.Start <= current.End + 6;

                if (isSameLabel && isAdjacent)
                {
                    current.End = Math.Max(current.End, next.End);
                    current.Text = originalText.Substring(current.Start, current.End - current.Start);
                }
                else
                {
                    merged.Add(CleanEntity(current));
                    current = next;
                }
            }

            merged.Add(CleanEntity(current));
            return merged;
        }

        private List<(int Start, int End)> GenerateOffsetsFromTokens(string originalText, uint[] tokens)
        {
            var offsets = new List<(int Start, int End)>();
            int currentSearchIndex = 0;

            foreach (uint tokenId in tokens)
            {
                string rawToken = _tokenizer.Decode(new uint[] { tokenId });

                // Clean up Byte-Pair Encoding markers, but preserve space structures
                string cleanChunk = rawToken.Replace("Ġ", " ").Replace("Ċ", "\n");

                if (string.IsNullOrEmpty(cleanChunk))
                {
                    offsets.Add((currentSearchIndex, currentSearchIndex));
                    continue;
                }

                // FIX: Match against the full chunk instead of .Trim() 
                // to prevent character coordinate shifting.
                int foundIdx = originalText.IndexOf(cleanChunk, currentSearchIndex, StringComparison.Ordinal);

                if (foundIdx != -1)
                {
                    int start = foundIdx;
                    int end = foundIdx + cleanChunk.Length;
                    offsets.Add((start, end));
                    currentSearchIndex = end;
                }
                else
                {
                    // Fallback try with trimmed data if structural characters mismatch
                    int fallbackIdx = originalText.IndexOf(cleanChunk.Trim(), currentSearchIndex, StringComparison.Ordinal);
                    if (fallbackIdx != -1)
                    {
                        int start = fallbackIdx;
                        int end = fallbackIdx + cleanChunk.Trim().Length;
                        offsets.Add((start, end));
                        currentSearchIndex = end;
                    }
                    else
                    {
                        offsets.Add((currentSearchIndex, currentSearchIndex));
                    }
                }
            }

            return offsets;
        }

        private static DetectedEntity CleanEntity(DetectedEntity ent)
        {
            string rawText = ent.Text;
            string strippedText = rawText.Trim();

            // Handle tracking leading spaces to compute index offsets safely
            int leadingSpaces = rawText.Length - rawText.TrimStart().Length;

            return new DetectedEntity
            {
                Start = ent.Start + leadingSpaces,
                End = ent.Start + leadingSpaces + strippedText.Length,
                Text = strippedText,
                Label = ent.Label
            };
        }

        private DenseTensor<long> CreateTensor(long[] data) => new DenseTensor<long>(data, new[] { 1, data.Length });

        private void InitModel(int Threads)
        {
            _session?.Dispose();
            _session = null;

            using var client = new HttpClient();
            client.DefaultRequestHeaders.Add("User-Agent", "OutSystems-PrivateRedactor-Plugin-ODC");

            var sessionOptions = new SessionOptions
            {
                IntraOpNumThreads = Threads,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableCpuMemArena = false,
                EnableMemoryPattern = false
            };

            try
            {
                string tempModelPath = Path.Combine(Path.GetTempPath(), "model.onnx");

                if (!File.Exists(tempModelPath))
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, "https://huggingface.co/bardsai/eu-pii-anonimization-multilang/resolve/main/onnx/model_quantized.onnx");
                    using var response = client.Send(request, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();

                    using var networkStream = response.Content.ReadAsStream();
                    using var fileStream = new FileStream(tempModelPath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, useAsync: false);

                    networkStream.CopyTo(fileStream);
                }

                _session = new InferenceSession(tempModelPath, sessionOptions);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Sequential initialization failed: {ex.Message}");
                throw;
            }
        }
    }
}