from datetime import UTC, datetime

from ai_service.rag.schemas import (
    BookmarkItem,
    BookmarkItemOut,
    CreateBookmarkIn,
)


def test_create_bookmark_in_with_camel_case() -> None:
    payload = {"sessionId": "sess-1", "turnIndex": 2, "note": "중요한 정보"}
    dto = CreateBookmarkIn.model_validate(payload)
    assert dto.session_id == "sess-1"
    assert dto.turn_index == 2
    assert dto.note == "중요한 정보"


def test_create_bookmark_in_with_optional_fields() -> None:
    payload = {"sessionId": "sess-2"}
    dto = CreateBookmarkIn.model_validate(payload)
    assert dto.session_id == "sess-2"
    assert dto.turn_index is None
    assert dto.note is None


def test_bookmark_item_out_serialization_alias() -> None:
    now = datetime.now(UTC)
    item = BookmarkItemOut(
        id="bm-1",
        sessionId="sess-1",
        turnIndex=1,
        title="세션 제목",
        content="답변 내용",
        note="메모",
        createdAt=now,
    )
    dumped = item.model_dump(by_alias=True)
    assert dumped["id"] == "bm-1"
    assert dumped["sessionId"] == "sess-1"
    assert dumped["turnIndex"] == 1
    assert dumped["title"] == "세션 제목"
    assert dumped["content"] == "답변 내용"
    assert dumped["note"] == "메모"
    assert dumped["createdAt"] == now


def test_bookmark_item_out_from_dict() -> None:
    now = datetime.now(UTC)
    data = {
        "_id": "bm-mongo-123",
        "sessionId": "sess-123",
        "turnIndex": 0,
        "title": "테스트 제목",
        "content": "답변",
        "note": "메모",
        "createdAt": now,
    }
    out = BookmarkItemOut.from_dict(data)
    assert out.id == "bm-mongo-123"
    assert out.session_id == "sess-123"
    assert out.turn_index == 0


def test_bookmark_domain_model() -> None:
    now = datetime.now(UTC)
    bm = BookmarkItem(
        id="bm-1",
        user_id="user-1",
        session_id="sess-1",
        turn_index=1,
        title="제목",
        content="내용",
        note="메모",
        created_at=now,
    )
    assert bm.id == "bm-1"
    assert bm.user_id == "user-1"
    out = BookmarkItemOut.from_domain(bm)
    assert out.id == "bm-1"
    assert out.session_id == "sess-1"
