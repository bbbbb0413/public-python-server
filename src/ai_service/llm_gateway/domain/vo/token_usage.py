from dataclasses import dataclass

from ai_service.shared_kernel.value_object import ValueObject


@dataclass(frozen=True)
class TokenUsageValue:
    prompt_tokens: int
    completion_tokens: int


class TokenUsage(ValueObject[TokenUsageValue]):
    def _validate(self, value: TokenUsageValue) -> None:
        if value.prompt_tokens < 0:
            raise ValueError("promptTokens는 0 이상의 정수여야 합니다.")
        if value.completion_tokens < 0:
            raise ValueError("completionTokens는 0 이상의 정수여야 합니다.")

    @classmethod
    def of(cls, prompt_tokens: int, completion_tokens: int) -> "TokenUsage":
        return cls(TokenUsageValue(prompt_tokens, completion_tokens))

    @property
    def prompt_tokens(self) -> int:
        return self.get_value().prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.get_value().completion_tokens

    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens
