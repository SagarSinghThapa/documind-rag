"""Automated unit tests for TextNormalizer."""

import pytest
from documind.ingestion.text_normalizer import TextNormalizer


class TestTextNormalizer:
    """Test suite for conservative text normalization."""

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    def test_empty_and_none_text(self, normalizer):
        """Empty or None text should return an empty string."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize(None) == ""

    def test_strip_null_and_control_characters(self, normalizer):
        """Null bytes and ASCII non-printable control characters should be removed."""
        raw = "Hello\x00 World!\x01\x02\x08 Test\x1f End"
        assert normalizer.normalize(raw) == "Hello World! Test End"

    def test_preserve_tabs_and_newlines_cleaned(self, normalizer):
        """Tabs should be normalized to single space, newlines preserved."""
        raw = "Line 1\twith\ttabs\nLine 2    with   spaces"
        expected = "Line 1 with tabs\nLine 2 with spaces"
        assert normalizer.normalize(raw) == expected

    def test_crlf_normalization(self, normalizer):
        """Windows CRLF and old Mac CR newlines should be standardized to LF."""
        raw = "Line 1\r\nLine 2\rLine 3\nLine 4"
        assert normalizer.normalize(raw) == "Line 1\nLine 2\nLine 3\nLine 4"

    def test_excessive_newlines_collapse_to_paragraph_break(self, normalizer):
        """Three or more consecutive newlines should collapse to a standard paragraph break (\n\n)."""
        raw = "Paragraph 1\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
        expected = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        assert normalizer.normalize(raw) == expected

    def test_hyphenated_linebreaks_conservative(self, normalizer):
        """Hyphenated words across line breaks should be merged only when safe."""
        raw = "This is a demon-\nstration of docu-\nment processing."
        expected = "This is a demonstration of document processing."
        assert normalizer.normalize(raw) == expected

    def test_preserve_technical_identifiers_and_numbers(self, normalizer):
        """Technical identifiers, hardware specs, decimal numbers, and hyphenated terms must be preserved."""
        raw = "Model: Intel Core i5-1135G7 @ 2.40GHz, RAM: 24GB.\nVersion: v2.1.0-alpha, Threshold: 0.005, Ratio: 16:9."
        expected = "Model: Intel Core i5-1135G7 @ 2.40GHz, RAM: 24GB.\nVersion: v2.1.0-alpha, Threshold: 0.005, Ratio: 16:9."
        assert normalizer.normalize(raw) == expected

    def test_preserve_punctuation_and_structure(self, normalizer):
        """Section headers, markdown bullet symbols, and punctuation must remain intact."""
        raw = "## 1. Executive Summary\n\n- Item A: High priority (Score: 95%)\n- Item B: Low priority (Score: 10%)"
        expected = "## 1. Executive Summary\n\n- Item A: High priority (Score: 95%)\n- Item B: Low priority (Score: 10%)"
        assert normalizer.normalize(raw) == expected

    def test_unicode_nfc_normalization(self, normalizer):
        """Composed and decomposed unicode characters should be normalized to NFC."""
        # 'e' + combining acute accent (\u0301) vs precomposed 'é' (\u00e9)
        decomposed = "resum\u0065\u0301"
        precomposed = "resum\u00e9"
        assert normalizer.normalize(decomposed) == precomposed
