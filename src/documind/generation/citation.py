"""Three-layer citation extraction, syntax validation, and metadata verification."""

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from documind.generation.schemas import Citation
from documind.retrieval.schemas import ContextReference

# Regex pattern matching deterministic source identifiers: [S1], [S2], [s1], etc.
SOURCE_ID_REGEX = re.compile(r"\[S(\d+)\]", re.IGNORECASE)

# Legacy regex pattern matching verbose inline citations: [Source: <filename>, p. <page>]
CITATION_REGEX = re.compile(
    r"\[Source:\s*([^,\]]+),\s*(?:pp?\.?|pages?)\s*(\d+(?:\s*-\s*\d+)?)\]",
    re.IGNORECASE,
)

# Common stopwords to ignore when checking evidence support
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "will", "shall", "that", "this", "these", "those", "it", "its", "as", "into",
    "source", "page", "pages", "document",
}


class CitationParser:
    """Extracts and verifies inline document citations against retrieved context references.
    
    Implements the 3-Layer Verification Model:
    - Layer 1 (Syntax): Validates bracketed citation syntax ([S1], [S2] or [Source: ...]).
    - Layer 2 (Metadata Linkage): Resolves cited source IDs or filenames strictly against
      trusted ContextReference records present in the retrieved context.
    - Layer 3 (Evidence Support): Validates that key factual terms in the claim appear
      in the referenced chunk text.
    """

    def __init__(
        self,
        citation_regex: Optional[re.Pattern] = None,
        source_id_regex: Optional[re.Pattern] = None,
    ) -> None:
        self._regex = citation_regex or CITATION_REGEX
        self._source_id_regex = source_id_regex or SOURCE_ID_REGEX

    @property
    def regex(self) -> re.Pattern:
        return self._regex

    @property
    def source_id_regex(self) -> re.Pattern:
        return self._source_id_regex

    def extract_citations(self, text: str) -> List[Citation]:
        """Parse all inline citations from generation output text (Layer 1)."""
        if not text:
            return []

        # Find matches along with their start index to maintain text order
        raw_matches: List[Tuple[int, Citation]] = []

        # 1. Parse deterministic source identifier tags ([S1], [S2], etc.)
        for match in self._source_id_regex.finditer(text):
            raw_text = match.group(0).strip()
            source_id = f"S{match.group(1)}"
            cit = Citation(
                raw_text=raw_text,
                document_name="",
                start_page=0,
                end_page=0,
                source_id=source_id,
                matched_chunk_id=None,
                is_syntax_valid=True,
                is_metadata_valid=False,
                is_evidence_supported=False,
            )
            raw_matches.append((match.start(), cit))

        # 2. Parse legacy verbose citations ([Source: doc.pdf, p. 1])
        for match in self._regex.finditer(text):
            raw_text = match.group(0).strip()
            doc_name = match.group(1).strip()
            page_part = match.group(2).strip()

            if "-" in page_part:
                parts = page_part.split("-")
                try:
                    start_page = int(parts[0].strip())
                    end_page = int(parts[1].strip())
                except ValueError:
                    continue
            else:
                try:
                    start_page = int(page_part)
                    end_page = start_page
                except ValueError:
                    continue

            cit = Citation(
                raw_text=raw_text,
                document_name=doc_name,
                start_page=start_page,
                end_page=end_page,
                source_id=None,
                matched_chunk_id=None,
                is_syntax_valid=True,
                is_metadata_valid=False,
                is_evidence_supported=False,
            )
            raw_matches.append((match.start(), cit))

        # Sort citations by appearance order in generated text
        raw_matches.sort(key=lambda x: x[0])
        return [c for _, c in raw_matches]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize document filename for robust matching."""
        base = os.path.basename(name.strip())
        return base.lower()

    @staticmethod
    def _extract_claim_sentence(text: str, citation_raw: str) -> str:
        """Locate the surrounding sentence containing the citation."""
        idx = text.find(citation_raw)
        if idx == -1:
            return text

        # Match true sentence boundaries (dot followed by whitespace, or newline)
        # Avoid splitting on dots inside filenames (e.g. FINZAA.pdf)
        pre_text = text[:idx]
        boundaries = [m.end() for m in re.finditer(r"(?:\.\s+|\n+)", pre_text)]
        start_boundary = max(boundaries) if boundaries else 0

        post_text = text[idx + len(citation_raw):]
        end_match = re.search(r"(?:\.|\n)", post_text)
        end_boundary = (idx + len(citation_raw) + end_match.end()) if end_match else len(text)

        sentence = text[start_boundary:end_boundary].replace(citation_raw, "").strip()
        return sentence

    @staticmethod
    def _check_evidence_overlap(sentence: str, chunk_text: str) -> bool:
        """Heuristic check: does chunk text contain meaningful content words from claim?"""
        tokens = re.findall(r"\b[A-Za-z0-9_]{3,}\b", sentence.lower())
        content_tokens = [t for t in tokens if t not in STOPWORDS]
        if not content_tokens:
            return True  # No specific content tokens to contradict

        chunk_lower = chunk_text.lower()
        matched = [t for t in content_tokens if t in chunk_lower]
        # At least 1 content token or 25% of tokens must appear in chunk
        return len(matched) >= min(1, len(content_tokens))

    def verify_citations(
        self,
        citations: List[Citation],
        references: List[ContextReference],
        answer_text: Optional[str] = None,
        chunk_text_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Citation], List[str]]:
        """Verify citations against structured ContextReference records (Layers 2 & 3).
        
        Args:
            citations: List of extracted Citation objects.
            references: Verifiable ContextReference objects from FormattedContext.
            answer_text: Optional full answer text to extract surrounding claim sentences.
            chunk_text_map: Optional mapping of chunk_id -> chunk text for Layer 3 validation.
            
        Returns:
            Tuple of (verified_citations, hallucinated_citation_strings).
        """
        verified_citations: List[Citation] = []
        hallucinated_citations: List[str] = []

        for cit in citations:
            matched_ref: Optional[ContextReference] = None

            # Case A: Citing deterministic source identifier [S1], [S2]
            if cit.source_id is not None:
                for ref in references:
                    if ref.source_id and ref.source_id.upper() == cit.source_id.upper():
                        matched_ref = ref
                        break

            # Case B: Legacy verbose citation [Source: doc.pdf, p. 1]
            elif cit.document_name:
                norm_cit_doc = self._normalize_name(cit.document_name)
                for ref in references:
                    norm_ref_doc = self._normalize_name(ref.document_name)
                    doc_matched = (
                        norm_cit_doc == norm_ref_doc
                        or norm_cit_doc in norm_ref_doc
                        or norm_ref_doc in norm_cit_doc
                    )
                    if not doc_matched:
                        continue

                    page_in_span = (
                        cit.start_page <= ref.end_page and cit.end_page >= ref.start_page
                    )
                    page_in_list = False
                    if ref.page_numbers:
                        cit_pages = set(range(cit.start_page, cit.end_page + 1))
                        page_in_list = bool(cit_pages.intersection(set(ref.page_numbers)))

                    if page_in_span or page_in_list:
                        matched_ref = ref
                        break

            # Check verification outcome
            if matched_ref is not None:
                evidence_supported = True
                if answer_text and chunk_text_map and matched_ref.chunk_id in chunk_text_map:
                    sentence = self._extract_claim_sentence(answer_text, cit.raw_text)
                    chunk_text = chunk_text_map[matched_ref.chunk_id]
                    evidence_supported = self._check_evidence_overlap(sentence, chunk_text)
                    # If sentence boundary was too narrow (e.g. multi-sentence answer with citation at end), check full answer
                    if not evidence_supported and len(answer_text) <= 500:
                        evidence_supported = self._check_evidence_overlap(answer_text, chunk_text)

                verified_cit = Citation(
                    raw_text=cit.raw_text,
                    document_name=matched_ref.document_name,
                    start_page=matched_ref.start_page,
                    end_page=matched_ref.end_page,
                    matched_chunk_id=matched_ref.chunk_id,
                    is_syntax_valid=True,
                    is_metadata_valid=True,
                    is_evidence_supported=evidence_supported,
                    source_id=matched_ref.source_id or cit.source_id,
                )
                verified_citations.append(verified_cit)
            else:
                unverified_cit = Citation(
                    raw_text=cit.raw_text,
                    document_name=cit.document_name or "",
                    start_page=cit.start_page,
                    end_page=cit.end_page,
                    matched_chunk_id=None,
                    is_syntax_valid=True,
                    is_metadata_valid=False,
                    is_evidence_supported=False,
                    source_id=cit.source_id,
                )
                verified_citations.append(unverified_cit)
                hallucinated_citations.append(cit.raw_text)

        return verified_citations, hallucinated_citations
