"""Paper ingestion MCP tool for research document processing.

Extracts text from PDFs (best-effort), parses coarse sections, derives
lightweight metadata, caches results, and provides summarization helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.core.proc import arun


class PaperIngestionTool:
    def __init__(self, papers_dir: Path | None = None) -> None:
        self.papers_dir = (papers_dir or Path("papers")).resolve()
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = (self.papers_dir / ".cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus: Any | None = None

    async def initialize(self, event_bus: Any, **_: Any) -> bool:
        self.event_bus = event_bus
        return True

    # --------------------------- Extraction ----------------------------- #
    async def extract_text_from_pdf(self, pdf_path: str) -> dict[str, Any]:
        try:
            target = (self.papers_dir / pdf_path).resolve()
            if not target.exists():
                return {"error": f"PDF file {pdf_path!r} not found"}

            cache_key = hashlib.md5(str(target).encode("utf-8")).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            text = await self._extract_pdf_text(target)
            if not text:
                return {"error": "Failed to extract text from PDF"}

            sections = self._parse_paper_sections(text)
            metadata = self._extract_metadata(text)
            result = {
                "text": text,
                "sections": sections,
                "metadata": metadata,
                "file_path": Path(pdf_path).as_posix(),
                "extraction_method": "best_effort",
            }
            cache_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return result
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"Failed to extract PDF text: {e}"}

    async def process_text_paper(
        self, text_content: str, title: str | None = None
    ) -> dict[str, Any]:
        try:
            sections = self._parse_paper_sections(text_content)
            metadata = self._extract_metadata(text_content)
            if title and metadata:
                metadata["title"] = title
            return {
                "text": text_content,
                "sections": sections,
                "metadata": metadata,
                "extraction_method": "text_processing",
            }
        except Exception as e:
            return {"error": f"Failed to process text: {e}"}

    # --------------------------- Search cache --------------------------- #
    async def search_papers(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        try:
            results: list[dict[str, Any]] = []
            for cf in self.cache_dir.glob("*.json"):
                try:
                    data = json.loads(cf.read_text(encoding="utf-8"))
                    searchable = []
                    meta = data.get("metadata") or {}
                    if meta:
                        searchable.append(meta.get("title") or "")
                        searchable.append(meta.get("abstract") or "")
                    searchable.append((data.get("text") or "")[:2000])
                    hay = "\n".join(searchable)
                    if query.lower() in hay.lower():
                        results.append(
                            {
                                "file_path": data.get("file_path"),
                                "title": (meta.get("title") or "Unknown"),
                                "relevance_snippet": self._extract_snippet(
                                    hay, query
                                ),
                            }
                        )
                except Exception:
                    continue
            return {
                "results": results[:max_results],
                "query": query,
                "count": len(results),
            }
        except Exception as e:
            return {"error": f"Failed to search papers: {e}"}

    # --------------------------- Download ------------------------------- #
    async def download_paper(
        self, url: str, filename: str | None = None
    ) -> dict[str, Any]:
        try:
            import aiohttp

            fn = filename or url.split("/")[-1]
            if not fn.lower().endswith(".pdf"):
                fn = f"{fn}.pdf"
            target = (self.papers_dir / fn).resolve()
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=60) as resp:
                    if resp.status != 200:
                        return {
                            "error": f"Failed to download: HTTP {resp.status}"
                        }
                    data = await resp.read()
                    target.write_bytes(data)
                    size = len(data)
            return {
                "success": True,
                "file_path": target.relative_to(self.papers_dir).as_posix(),
                "size": size,
                "url": url,
            }
        except Exception as e:
            return {"error": f"Failed to download paper: {e}"}

    # --------------------------- Summary -------------------------------- #
    async def generate_paper_summary(
        self, pdf_path: str, focus_areas: list[str] | None = None
    ) -> dict[str, Any]:
        try:
            data = await self.extract_text_from_pdf(pdf_path)
            if "error" in data:
                return data
            sections = data.get("sections") or []
            meta = data.get("metadata") or {}
            foci = [
                s.lower()
                for s in (
                    focus_areas
                    or ["abstract", "methods", "results", "conclusion"]
                )
            ]
            key_sections = [
                s
                for s in sections
                if (s.get("section_type") or "").lower() in foci
            ]
            return {
                "title": meta.get("title") or "Unknown",
                "authors": meta.get("authors") or [],
                "key_sections": key_sections,
                "abstract": meta.get("abstract") or "",
                "keywords": meta.get("keywords") or [],
                "algorithmic_content": self._extract_algorithmic_content(
                    sections
                ),
            }
        except Exception as e:
            return {"error": f"Failed to generate summary: {e}"}

    # --------------------------- Internals ------------------------------ #
    async def _extract_pdf_text(self, pdf_path: Path) -> str | None:
        # Try PyPDF2
        try:  # pragma: no cover - optional dep
            import PyPDF2

            text = []
            with pdf_path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        text.append(page.extract_text() or "")
                    except Exception:
                        continue
            joined = "\n".join(text).strip()
            if joined:
                return joined
        except Exception:
            pass

        # Try pdfplumber
        try:  # pragma: no cover - optional dep
            import pdfplumber

            text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        text.append(page.extract_text() or "")
                    except Exception:
                        continue
            joined = "\n".join(text).strip()
            if joined:
                return joined
        except Exception:
            pass

        # Fallback to system pdftotext if available
        try:
            out = await arun(["pdftotext", str(pdf_path), "-"], timeout=60)
            if out and out.strip():
                return out
        except Exception:
            pass
        return None

    def _parse_paper_sections(self, text: str) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        patterns: dict[str, str] = {
            "abstract": r"(?i)\babstract\b",
            "introduction": r"(?i)\b(introduction|1\.[\s]*introduction)\b",
            "methods": r"(?i)\b(methods|methodology|approach)\b",
            "results": r"(?i)\b(results|experiments|evaluation)\b",
            "conclusion": r"(?i)\b(conclusion|conclusions)\b",
        }
        current: str | None = None
        buf: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            matched: str | None = None
            for k, pat in patterns.items():
                if re.search(pat, line):
                    matched = k
                    break
            if matched:
                if current and buf:
                    sections.append(
                        {
                            "title": current,
                            "content": "\n".join(buf),
                            "section_type": current,
                        }
                    )
                current = matched
                buf = []
            elif current:
                buf.append(line)
        if current and buf:
            sections.append(
                {
                    "title": current,
                    "content": "\n".join(buf),
                    "section_type": current,
                }
            )
        return sections

    def _extract_metadata(self, text: str) -> dict[str, Any]:
        lines = [l.strip() for l in text.splitlines()[:80] if l.strip()]
        title = next(
            (l for l in lines if len(l) > 20 and not l.isupper()),
            "Unknown Title",
        )
        abstract_match = re.search(
            r"(?is)abstract\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\s*1\.|introduction)\s*",
            text,
        )
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        kw_match = re.search(
            r"(?i)keywords?\s*[:\-]?\s*(.+?)\s*(?=\n\s*\n)", text
        )
        keywords = []
        if kw_match:
            keywords = [
                k.strip() for k in kw_match.group(1).split(",") if k.strip()
            ]
        return {
            "title": title,
            "authors": [],
            "abstract": abstract,
            "keywords": keywords,
        }

    def _extract_snippet(
        self, hay: str, needle: str, max_len: int = 200
    ) -> str:
        h, n = hay.lower(), needle.lower()
        i = h.find(n)
        if i < 0:
            return (hay[:max_len] + "...") if len(hay) > max_len else hay
        start = max(0, i - max_len // 2)
        end = min(len(hay), i + max_len // 2)
        snip = hay[start:end]
        if start > 0:
            snip = "..." + snip
        if end < len(hay):
            snip = snip + "..."
        return snip

    def _extract_algorithmic_content(
        self, sections: list[dict[str, Any]]
    ) -> list[str]:
        pats = [
            r"(?i)algorithm\s+\d+",
            r"(?i)procedure\s+\w+",
            r"(?i)function\s+\w+",
            r"(?i)input:\s*.+",
            r"(?i)output:\s*.+",
            r"(?i)for\s+each\s+.+",
            r"(?i)while\s+.+",
            r"(?i)if\s+.+\s+then",
        ]
        out: set[str] = set()
        for sec in sections:
            body = sec.get("content") or ""
            for pat in pats:
                for m in re.findall(pat, body, flags=re.MULTILINE):
                    out.add(m)
        return sorted(out)
