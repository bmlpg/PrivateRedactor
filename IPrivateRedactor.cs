using OutSystems.ExternalLibraries.SDK;

namespace PrivateRedactor
{
    [OSInterface(Name = "PrivateRedactor", Description = "Free and fully local AI PII redactor for ODC.", IconResourceName = "PrivateRedactor.resources.privateredactor_logo.png")]
    public interface IPrivateRedactor
    {

        [OSAction(ReturnName = "Result", Description = "Redacts sensitive text locally using a token-classification AI model to strip out personally identifiable information (PII).")]
        public RedactionResult Redact(
            [OSParameter(Description = "The raw source text to process for PII identification and redaction.")]
            string Text,
            [OSParameter(Description = "Comma-separated list of entity types to detect and redact. If left empty all the available entity types are detected and redacted.")]
            string DetectableEntitiesOverride = "",
            [OSParameter(Description = "Number of CPU cores to use for math operations. Ensure it is at least 1. If set higher than the host’s logical cores, performance will actually degrade. Default 1.")]
            int Threads = 1
        );
    }
}