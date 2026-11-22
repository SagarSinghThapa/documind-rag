"""Conservative text normalization for extracted PDF text.

Preserves paragraph boundaries, technical identifiers, punctuation, and structural headings
while cleaning extraction artifacts, null bytes, and erratic whitespace.
"""

import re
import unicodedata


class TextNormalizer:
    """Performs conservative normalization on extracted document text."""

    def __init__(self, fix_hyphenated_linebreaks: bool = True):
        self.fix_hyphenated_linebreaks = fix_hyphenated_linebreaks

    def normalize(self, text: str) -> str:
        """Normalize extracted plain text conservatively.
        
        Args:
            text: Raw extracted text string.
            
        Returns:
            Normalized text string.
        """
        if not text:
            return ""

        # 1. Unicode NFC normalization (ensures consistent composed characters)
        text = unicodedata.normalize("NFC", text)

        # 2. Strip null bytes and non-printable control characters (preserve \n and \t)
        # Keep ASCII characters >= 32, plus \n (10) and \t (9)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # 3. Standardize newline representations (CRLF and CR -> LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 4. Optional conservative de-hyphenation across linebreaks
        # Matches lowercase letter + hyphen + newline + lowercase letter (e.g. "con-\nnection" -> "connection")
        # Carefully avoids compound numbers (e.g. "11-\n35") or uppercase acronyms
        if self.fix_hyphenated_linebreaks:
            text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

        # 5. Clean horizontal whitespace per line (tabs and spaces -> single space)
        # Process line by line to preserve exact line break structures
        lines = []
        for line in text.split("\n"):
            # Replace multiple horizontal spaces or tabs with a single space
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)

        text = "\n".join(lines)

        # 6. Consolidate excessive vertical newlines: 3+ consecutive newlines become 2 (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 7. Final outer strip
        return text.strip()
