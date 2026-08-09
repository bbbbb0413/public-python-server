class PromptTemplateNotFoundError(Exception):
    def __init__(self, name: str, version: int) -> None:
        super().__init__(f"프롬프트를 찾을 수 없습니다: {name} v{version}")
        self.name = name
        self.version = version
