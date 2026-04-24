
import os
from google.oauth2 import service_account

from src.utils.llm.models.data import ModelResponse


# Required scopes for Vertex AI
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/cloud-platform.read-only"
]

# Map of Claude model names to their Vertex AI model names
claude_vertex_model_mapping = {
    "claude-3-haiku": "claude-3-haiku@20240307",
    "claude-3-sonnet": "claude-3-sonnet@20240229",
    "claude-3-opus": "claude-3-opus@20240229",
    "claude-3-5-haiku": "claude-3-5-haiku@20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-v2@20241022",
    "claude-3-7-sonnet": "claude-3-7-sonnet@20250219"
}

class ClaudeVertexEngine:
    """
    A wrapper class for the AnthropicVertex client to make it compatible
    with the existing engine interface.
    """
    def __init__(self, model_name: str, **kwargs):
        try:
            from anthropic import AnthropicVertex
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required to use Claude models. "
                "Please install it with 'pip install anthropic'."
            )
        
        self.model_name = model_name
        self.vertex_model_name = \
              claude_vertex_model_mapping.get(model_name, model_name)
        self.kwargs = kwargs
        
        # Get GCP configuration from environment variables
        project_id = os.getenv("GCP_PROJECT")
        region = os.getenv("GCP_REGION")
        credentials_path = os.getenv("GCP_CREDENTIALS")
        
        if not project_id or not region:
            raise ValueError(
                "GCP_PROJECT and GCP_REGION must be provided in .env"
            )

        # Use key file if available, otherwise fall back to Application Default
        # Credentials (e.g. Cloud Run service account, `gcloud auth application-default`)
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=SCOPES
            )
        else:
            credentials = None  # AnthropicVertex will use ADC automatically

        # Initialize the AnthropicVertex client
        try:
            self.client = AnthropicVertex(
                project_id=project_id,
                region=region,
                **({"credentials": credentials} if credentials is not None else {})
            )
        except Exception as e:
            print(f"Warning: Could not initialize Claude models: {e}")
            print("Please request access to Claude models in your GCP project.")
            print("Falling back to default model...")
            # You could implement a fallback here
            raise ValueError(
                f"Your GCP project ({project_id}) does not"
                f" have access to Claude models. "
            )
    
    def invoke(self, prompt, **kwargs) -> ModelResponse:
        """
        Invoke the Claude model with the given prompt.

        Args:
            prompt: The input prompt as a string
            **kwargs: Additional keyword arguments for the model invocation

        Returns:
            A ModelResponse object with the model's response and token usage
        """
        # Convert string prompt to message format
        messages = [{"role": "user", "content": prompt}]

        # Create the message with the Claude model
        response = self.client.messages.create(
            model=self.vertex_model_name,
            messages=messages,
            **self.kwargs,
            **kwargs
        )

        # Extract content from Claude response
        content = response.content[0].text if response.content else ""

        # Create ModelResponse with content and usage metadata
        model_response = ModelResponse(content)

        # Extract token usage from Claude response
        if hasattr(response, 'usage'):
            model_response.response_metadata = {
                'token_usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                }
            }

        return model_response


class ClaudeDirectEngine:
    """Calls the Anthropic API directly (for Claude 4.x and newer models)."""

    def __init__(self, model_name: str, **kwargs):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install the 'anthropic' package: pip install anthropic")

        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens_to_sample", 4096)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.client = Anthropic(api_key=api_key)

    def invoke(self, prompt, **kwargs) -> ModelResponse:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text if response.content else ""
        model_response = ModelResponse(content)
        if hasattr(response, "usage"):
            model_response.response_metadata = {
                "token_usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            }
        return model_response