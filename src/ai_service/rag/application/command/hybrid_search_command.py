from dataclasses import dataclass


@dataclass(frozen=True)
class HybridSearchCommand:
    question: str
    top_k: int = 5
    use_hyde: bool = False

    @property
    def query(self) -> str:
        return self.question


__all__ = ["HybridSearchCommand"]
