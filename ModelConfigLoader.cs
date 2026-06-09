using System.Text.Json;

namespace PrivateRedactor
{
    public static class ModelConfigLoader
    {
        /// <summary>
        /// Loads the id2label dictionary from the model's config.json file.
        /// </summary>
        /// <param name="configPath">The absolute path to the config.json file.</param>
        /// <returns>A dictionary mapping class IDs (int) to label names (string).</returns>
        public static Dictionary<int, string> LoadId2Label(string configPath)
        {
            var id2LabelMap = new Dictionary<int, string>();

            try
            {
                if (!File.Exists(configPath))
                {
                    throw new FileNotFoundException($"Model configuration file not found at: {configPath}");
                }

                string jsonContent = File.ReadAllText(configPath);

                // Use the built-in System.Text.Json to parse the file efficiently
                using JsonDocument doc = JsonDocument.Parse(jsonContent);
                JsonElement root = doc.RootElement;

                // Look for the standard Hugging Face "id2label" property
                if (root.TryGetProperty("id2label", out JsonElement id2LabelElement))
                {
                    foreach (JsonProperty property in id2LabelElement.EnumerateObject())
                    {
                        // Hugging Face JSON keys are strings (e.g., "0"), but we parse them to int IDs
                        if (int.TryParse(property.Name, out int id))
                        {
                            string label = property.Value.GetString() ?? "O";
                            id2LabelMap[id] = label;
                        }
                    }
                }
                else
                {
                    throw new KeyNotFoundException("The 'id2label' key was not found in the config.json file.");
                }
            }
            catch (Exception ex)
            {
                // Fallback / standard error escalation so your initialization log catches it
                throw new InvalidOperationException($"Failed to load model labels from config: {ex.Message}", ex);
            }

            return id2LabelMap;
        }
    }
}