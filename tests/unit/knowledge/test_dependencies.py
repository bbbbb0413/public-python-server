from unittest.mock import MagicMock

from fastapi import Request

from ai_service.knowledge.application.ingest_document_use_case import IngestDocumentUseCase
from ai_service.knowledge.dependencies import get_ingest_use_case


def test_get_ingest_use_case_returns_use_case_from_app_state() -> None:
    mock_request = MagicMock(spec=Request)
    mock_use_case = MagicMock(spec=IngestDocumentUseCase)
    mock_request.app.state.ingest_use_case = mock_use_case

    result = get_ingest_use_case(mock_request)

    assert result is mock_use_case
