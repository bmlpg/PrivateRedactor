using OutSystems.ExternalLibraries.SDK;

namespace PrivateRedactor
{
    [OSStructure(Description = "Represents a personally identifiable information (PII) entity identified within the text.")]
    public struct DetectedEntity
    {
        [OSStructureField(DataType = OSDataType.Integer, Description = "The zero-based starting character index of the detected entity in the original text.", IsMandatory = false)]
        public int Start;
        [OSStructureField(DataType = OSDataType.Integer, Description = "The zero-based ending character index (exclusive) of the detected entity in the original text.", IsMandatory = false)]
        public int End;
        [OSStructureField(DataType = OSDataType.Text, Description = "The category or type of PII detected (e.g., FIRST_NAME, EMAIL, STREET_ADDRESS).", IsMandatory = false)]
        public string Label;
        [OSStructureField(DataType = OSDataType.Text, Description = "The original, unredacted text segment that triggered the classification.", IsMandatory = false)]
        public string Text;
    }

    [OSStructure(Description = "The output payload containing the anonymized results and execution telemetry.")]
    public struct RedactionResult
    {
        [OSStructureField(DataType = OSDataType.Text, Description = "The processed version of the text where identified PII entities have been replaced with their respective mask tags.", IsMandatory = false)]
        public string RedactedText;
        [OSStructureField(DataType = OSDataType.InferredFromDotNetType, Description = "A collection of all individual PII entities detected during the token classification pass.", IsMandatory = false)]
        public List<DetectedEntity> Entities;
        [OSStructureField(DataType = OSDataType.LongInteger, Description = "The execution time taken by the ONNX inference engine and pipeline logic, measured in milliseconds.", IsMandatory = false)]
        public long Duration;
    }
}