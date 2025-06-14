import tempfile
from pathlib import Path

from ragsdk.core.sources.local import LocalFileSource  # type: ignore
from ragsdk.document_search.documents.document import (  # type: ignore
    DocumentMeta,
    DocumentType,
    TextDocument,
)


async def test_loading_local_file_source():
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"test")
        f.seek(0)

        source = LocalFileSource(path=Path(f.name))

        document_meta = DocumentMeta(document_type=DocumentType.TXT, source=source)

        document = await document_meta.fetch()

        assert isinstance(document, TextDocument)
        assert document.content == "test"
