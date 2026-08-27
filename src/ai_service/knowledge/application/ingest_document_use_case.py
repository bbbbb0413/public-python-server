import hashlib
import io
import logging
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ai_service.knowledge.repository import DocumentRepository, QdrantVectorAdapter
from ai_service.knowledge.schemas import (
    Chunk,
    Document,
    IngestDocumentCommand,
    VectorDocument,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.schemas import LlmMessage
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator

logger = logging.getLogger(__name__)

KOREAN_SENTENCE_ENDINGS = [
    "다. ",
    "요. ",
    "까. ",
    "죠. ",
    "나. ",
    "데. ",
    "네. ",
    "군. ",
    "음. ",
    "지. ",
    "야. ",
    "아. ",
    "어. ",
    "고. ",
    "며. ",
    "고요. ",
    "네요. ",
    "데요. ",
]

# Parent chunks(1536자) — LLM 컨텍스트 전달용
PARENT_CHUNK_SIZE = 1536
PARENT_CHUNK_OVERLAP = 300
# Child chunks(512자) — 벡터 검색용(정밀 매칭)
CHILD_CHUNK_SIZE = 512
CHILD_CHUNK_OVERLAP = 100

CONTEXT_PREFIX_LOOKBACK = 300
CONTEXT_PREFIX_LOOKAHEAD = 100
PARENT_ANCHOR_LENGTH = 100
DOC_SAMPLE_LENGTH = 200
OFFSET_PROBE_LENGTH = 50


class IngestDocumentUseCase:
    def __init__(
        self,
        document_repo: DocumentRepository | Any,
        vector_store: QdrantVectorAdapter | Any,
        embedding_provider: Any,
        llm_provider: Any,
        rag_validator: RagContentValidator,
        contextual_embeddings_enabled: bool = False,
    ) -> None:
        self._document_repo = document_repo
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._rag_validator = rag_validator
        self._contextual_embeddings_enabled = contextual_embeddings_enabled

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            separators=["\f", "\n\n", "\n", *KOREAN_SENTENCE_ENDINGS, " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", *KOREAN_SENTENCE_ENDINGS, " ", ""],
        )

    async def execute(self, command: IngestDocumentCommand) -> Document:
        document: Document
        if command.document_id:
            existing = await self._document_repo.find_by_id(command.document_id)
            document = existing or Document.create(command.file_name, command.mime_type)
        else:
            document = Document.create(command.file_name, command.mime_type)
            document = await self._document_repo.persist(document)

        document_id = document.id
        assert document_id is not None

        try:
            raw_text = self._extract_text(command.content, command.mime_type)
            verdict = self._rag_validator.inspect_input(raw_text)
            if not verdict.is_allowed():
                raise ValueError(f"인제스트 차단: {verdict.get_reason()}")

            vector_docs = await self._build_vector_docs(raw_text, document_id, command.file_name)

            await self._vector_store.delete_by_document_id(document_id)
            await self._vector_store.upsert(vector_docs)

            processed = document.mark_processed(len(vector_docs))
            return await self._document_repo.update(processed)
        except Exception:
            logger.exception("문서 수집 실패: %s", command.file_name)
            failed = document.mark_failed()
            await self._document_repo.update(failed)
            raise

    async def _build_vector_docs(
        self, raw_text: str, document_id: str, file_name: str
    ) -> list[VectorDocument]:
        parent_texts = self._split_by_paragraph_first(raw_text)
        parent_offsets = self._compute_chunk_offsets(raw_text, parent_texts)

        if self._contextual_embeddings_enabled:
            context_prefixes = [
                await self._generate_contextual_prefix(
                    file_name, raw_text, parent_text, parent_offsets[i]
                )
                for i, parent_text in enumerate(parent_texts)
            ]
        else:
            context_prefixes = ["" for _ in parent_texts]

        vector_docs: list[VectorDocument] = []
        global_child_index = 0

        for pi, parent_text in enumerate(parent_texts):
            parent_id = hashlib.sha256(f"{document_id}:parent:{pi}".encode()).hexdigest()
            context_prefix = context_prefixes[pi]

            child_texts = self._child_splitter.split_text(parent_text)

            # contextual prefix가 없을 때도 parent 앞 100자를 붙여 회사명/섹션 헤더가
            # 모든 child 벡터에 포함되도록 함
            parent_anchor = parent_text[:PARENT_ANCHOR_LENGTH].strip()
            embedding_inputs = [
                (
                    f"{context_prefix or parent_anchor}\n\n{ct}"
                    if (context_prefix or parent_anchor)
                    else ct
                )
                for ct in child_texts
            ]
            embeddings = await self._embedding_provider.embed(embedding_inputs)

            for ci, child_text in enumerate(child_texts):
                chunk = Chunk.of(
                    child_text,
                    global_child_index,
                    document_id,
                    char_count=len(child_text),
                    parent_chunk_id=parent_id,
                )

                vector_docs.append(
                    VectorDocument(
                        id=hashlib.sha256(
                            f"{document_id}:child:{global_child_index}".encode()
                        ).hexdigest(),
                        text=chunk.get_text(),
                        embedding=embeddings[ci],
                        metadata=VectorDocumentMetadata(
                            document_id=document_id,
                            file_name=file_name,
                            chunk_index=chunk.get_index(),
                            char_count=chunk.get_char_count(),
                            parent_text=parent_text,
                            parent_chunk_id=parent_id,
                        ),
                    )
                )
                global_child_index += 1

        return vector_docs

    async def _generate_contextual_prefix(
        self, doc_title: str, full_text: str, chunk_text: str, chunk_offset: int
    ) -> str:
        context_start = max(0, chunk_offset - CONTEXT_PREFIX_LOOKBACK)
        context_end = min(len(full_text), chunk_offset + len(chunk_text) + CONTEXT_PREFIX_LOOKAHEAD)
        local_context = full_text[context_start:context_end]

        mid = len(full_text) // 2
        doc_sample = (
            f"{full_text[:DOC_SAMPLE_LENGTH]}\n...\n" f"{full_text[mid : mid + DOC_SAMPLE_LENGTH]}"
        )

        messages = [
            LlmMessage(
                role="user",
                content=f"""<document>
제목: {doc_title}
대표 내용: {doc_sample}
</document>

위 문서에서 아래 청크의 핵심 내용을 1문장으로 설명하세요. 설명 문장만 출력하세요.

<chunk>{local_context}</chunk>""",
            )
        ]

        tokens = [token async for token in self._llm_provider.stream(messages)]
        return "".join(tokens).strip()

    def _split_by_paragraph_first(self, text: str) -> list[str]:
        segments = [s for s in re.split(r"\f|\n\n", text) if s.strip()]
        coarse_chunks: list[str] = []
        current = ""

        for seg in segments:
            joined = f"{current}\n\n{seg}" if current else seg
            if len(joined) > PARENT_CHUNK_SIZE and current:
                coarse_chunks.append(current.strip())
                current = seg
            else:
                current = joined
        if current:
            coarse_chunks.append(current.strip())

        result: list[str] = []
        for chunk in coarse_chunks:
            if len(chunk) > PARENT_CHUNK_SIZE:
                result.extend(self._parent_splitter.split_text(chunk))
            else:
                result.append(chunk)

        return result if result else self._parent_splitter.split_text(text)

    @staticmethod
    def _compute_chunk_offsets(full_text: str, chunks: list[str]) -> list[int]:
        offsets: list[int] = []
        search_start = 0
        for chunk in chunks:
            probe = chunk[:OFFSET_PROBE_LENGTH]
            idx = full_text.find(probe, search_start)
            offset = search_start if idx == -1 else idx
            offsets.append(offset)
            if idx != -1:
                search_start = idx + len(chunk)
        return offsets

    def _extract_text(self, content: bytes, mime_type: str) -> str:
        if mime_type == "application/pdf":
            reader = PdfReader(io.BytesIO(content))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return self._clean_pdf_text("\f".join(pages_text))
        return content.decode("utf-8")

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        # \f(form feed)는 PDF 페이지 경계 구분자로 보존한다.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
        text = re.sub(r"([가-힣A-Za-z0-9,\.。\)\]'\"])\n([가-힣A-Za-z0-9\(\['\"])", r"\1 \2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
