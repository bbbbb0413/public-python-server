from ai_service.rag.schemas import Critique


class QueryRefinerService:
    def refine(self, original_question: str, critique: Critique) -> str:
        next_query = critique.get_next_query()
        if next_query and next_query.strip():
            return next_query

        missing = critique.get_missing()
        if missing:
            return f"{original_question} (특히: {', '.join(missing)})"

        return original_question


__all__ = ["QueryRefinerService"]
