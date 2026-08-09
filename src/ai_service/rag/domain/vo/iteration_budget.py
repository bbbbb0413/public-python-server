from dataclasses import dataclass

from ai_service.shared_kernel.value_object import ValueObject


@dataclass(frozen=True)
class IterationBudgetProps:
    max_iterations: int
    token_budget: int
    timeout_ms: int


class IterationBudget(ValueObject[IterationBudgetProps]):
    def _validate(self, value: IterationBudgetProps) -> None:
        if value.max_iterations < 1 or value.max_iterations > 10:
            raise ValueError("max_iterations는 1 이상 10 이하여야 합니다.")
        if value.token_budget <= 0:
            raise ValueError("token_budget은 양수여야 합니다.")
        if value.timeout_ms <= 0:
            raise ValueError("timeout_ms는 양수여야 합니다.")

    def is_exhausted(self, iterations_completed: int, tokens_used: int, elapsed_ms: float) -> bool:
        value = self.get_value()
        return (
            iterations_completed >= value.max_iterations
            or tokens_used >= value.token_budget
            or elapsed_ms >= value.timeout_ms
        )

    def get_max_iterations(self) -> int:
        return self.get_value().max_iterations

    def get_token_budget(self) -> int:
        return self.get_value().token_budget

    def get_timeout_ms(self) -> int:
        return self.get_value().timeout_ms

    @classmethod
    def of(cls, max_iterations: int, token_budget: int, timeout_ms: int) -> "IterationBudget":
        return cls(IterationBudgetProps(max_iterations, token_budget, timeout_ms))
