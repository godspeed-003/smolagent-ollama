import requests

class OllamaModel:
    def __init__(self, model_name="llama3.1:latest", api_url="http://localhost:11434/api/generate", max_tokens=2048, temperature=0.7):
        self.model_name = model_name
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt, **kwargs):
        # If prompt is a list, flatten it to a string
        if isinstance(prompt, list):
            def extract_text(msg):
                if isinstance(msg, dict):
                    content = msg.get('content')
                else:
                    content = getattr(msg, 'content', '')
                if isinstance(content, list):
                    return ' '.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in content])
                return str(content)
            prompt_str = '\n'.join([extract_text(m) for m in prompt])
        else:
            prompt_str = str(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt_str,
            "stream": False
        }
        # Print a concise payload (truncate prompt if too long)
        prompt_preview = (payload['prompt'][:100] + '...') if len(payload['prompt']) > 100 else payload['prompt']
        print(f"OLLAMA PAYLOAD: model={payload['model']}, prompt='{prompt_preview}'")
        response = requests.post(self.api_url, json=payload)
        print(f"OLLAMA STATUS: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        # Print only the response text (not the full JSON or context array)
        response_text = result.get('response', '')
        print(f"OLLAMA RESPONSE: {response_text}")

        # Fallback ChatMessage and MessageRole definitions for compatibility
        class ChatMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content
        class MessageRole:
            ASSISTANT = "assistant"
        return ChatMessage(role=MessageRole.ASSISTANT, content=response_text)

    # Optional: Add attributes for compatibility with smolagents if needed
    last_input_token_count = None
    last_output_token_count = None
