from datetime import UTC, datetime

from ai_service.knowledge.schemas import Document, DocumentOut, DocumentProps


def test_document_props_default_error_message():
    props = DocumentProps(
        file_name="test.txt",
        mime_type="text/plain",
        status="pending",
        chunk_count=0,
    )
    assert props.error_message is None


def test_document_restore_with_error_message():
    created_at = datetime.now(UTC)
    props = DocumentProps(
        id="doc-1",
        file_name="test.txt",
        mime_type="text/plain",
        status="failed",
        chunk_count=0,
        created_at=created_at,
        error_message="인제스트 차단: 악성 프롬프트 감지",
    )
    doc = Document.restore(props)
    assert doc.error_message == "인제스트 차단: 악성 프롬프트 감지"
    assert doc.status == "failed"


def test_document_mark_failed_stores_error_message():
    doc = Document.create("sample.txt", "text/plain")
    failed = doc.mark_failed("PDF 파싱 에러")
    assert failed.status == "failed"
    assert failed.chunk_count == 0
    assert failed.error_message == "PDF 파싱 에러"


def test_document_mark_processed_clears_error_message():
    doc = Document.create("sample.txt", "text/plain")
    failed = doc.mark_failed("임시 네트워크 오류")
    assert failed.error_message == "임시 네트워크 오류"

    processed = failed.mark_processed(5)
    assert processed.status == "processed"
    assert processed.chunk_count == 5
    assert processed.error_message is None


def test_document_out_serialization_with_error_message():
    doc = Document(
        id="doc-123",
        file_name="doc.pdf",
        mime_type="application/pdf",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
        error_message="지원하지 않는 압축 포맷",
    )
    out = DocumentOut.from_domain(doc)
    assert out.error_message == "지원하지 않는 압축 포맷"

    dumped = out.model_dump(by_alias=True)
    assert dumped["errorMessage"] == "지원하지 않는 압축 포맷"


def test_document_out_serialization_without_error_message():
    doc = Document.create("doc.pdf", "application/pdf")
    out = DocumentOut.from_domain(doc)
    assert out.error_message is None

    dumped = out.model_dump(by_alias=True)
    assert dumped["errorMessage"] is None
