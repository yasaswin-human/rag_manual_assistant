import requests
import json
from src.config import Config

class OllamaLLM:
    def __init__(self, model=Config.OLLAMA_MODEL, base_url=Config.OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Send prompt to Ollama and return response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Error: Unable to reach LLM. Please check Ollama is running."

# Singleton
llm = OllamaLLM()