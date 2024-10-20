import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class RequestWrapper:
    prompt: str
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    timestamp: float = field(default_factory=time.time)
    max_length: int = 100
    num_return_sequences: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    stop_sequences: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "timestamp": self.timestamp,
            "max_length": self.max_length,
            "num_return_sequences": self.num_return_sequences,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestWrapper':
        return cls(**data)

    def update_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        return self.metadata.get(key)

# Example usage
if __name__ == "__main__":
    # Create a new request
    request = RequestWrapper(
        prompt="Translate the following English text to French: 'Hello, how are you?'",
        max_length=50,
        temperature=0.7,
        metadata={"source": "user_input", "priority": "high"}
    )

    # Print the request details
    print(f"Request ID: {request.request_id}")
    print(f"Prompt: {request.prompt}")
    print(f"Timestamp: {request.timestamp}")
    print(f"Metadata: {request.metadata}")

    # Update metadata
    request.update_metadata("processing_started", time.time())

    # Convert to dictionary and back
    request_dict = request.to_dict()
    reconstructed_request = RequestWrapper.from_dict(request_dict)

    print("\nReconstructed Request:")
    print(f"Request ID: {reconstructed_request.request_id}")
    print(f"Prompt: {reconstructed_request.prompt}")
    print(f"Metadata: {reconstructed_request.metadata}")
