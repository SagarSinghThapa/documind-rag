"""Grounding evaluation engine measuring factual support, citation accuracy, and abstention."""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional

from documind.generation.generator import GroundedGenerator
from documind.generation.schemas import GroundedAnswer


@dataclass(frozen=True)
class GroundingMetrics:
    """Formal evaluation metrics measuring grounding faithfulness and latency."""
    total_queries: int
    positive_queries: int
    negative_queries: int
    answer_correctness: float                   # Accuracy on Q+ against verified excerpts
    evidence_support_rate: float                # Q+ answers with verified or partially_supported status
    citation_validity_rate: float               # Valid citations / Total citations
    citation_support_accuracy: float            # Evidence-supported valid citations / Total valid citations
    negative_abstention_rate: float             # Abstention recall on Q- (Target: >= 87.5%)
    abstention_precision: float                 # True negative abstentions / Total abstentions triggered
    abstention_recall: float                    # True negative abstentions / Total Q-
    unsupported_answer_rate: float              # Answers with ungrounded claims
    false_answer_rate: float                    # Q- where model failed to abstain
    abstention_errors: float                    # Q+ where model abstained erroneously
    mean_generation_latency_ms: float
    mean_total_latency_ms: float
    mean_tokens_per_sec: float
    query_evaluations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "positive_queries": self.positive_queries,
            "negative_queries": self.negative_queries,
            "answer_correctness": round(self.answer_correctness, 4),
            "evidence_support_rate": round(self.evidence_support_rate, 4),
            "citation_validity_rate": round(self.citation_validity_rate, 4),
            "citation_support_accuracy": round(self.citation_support_accuracy, 4),
            "negative_abstention_rate": round(self.negative_abstention_rate, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "abstention_recall": round(self.abstention_recall, 4),
            "unsupported_answer_rate": round(self.unsupported_answer_rate, 4),
            "false_answer_rate": round(self.false_answer_rate, 4),
            "abstention_errors": round(self.abstention_errors, 4),
            "mean_generation_latency_ms": round(self.mean_generation_latency_ms, 2),
            "mean_total_latency_ms": round(self.mean_total_latency_ms, 2),
            "mean_tokens_per_sec": round(self.mean_tokens_per_sec, 2),
        }


class GroundingEvaluator:
    """Evaluates GroundedGenerator against ground-truth query datasets."""

    def __init__(
        self,
        generator: GroundedGenerator,
        eval_data_path: str = "data/evaluation/retrieval_ground_truth.json",
    ) -> None:
        self._generator = generator
        self._eval_data_path = eval_data_path

    def _load_data(self, split: str) -> List[Dict[str, Any]]:
        """Load ground-truth queries filtered by dataset split."""
        if not os.path.exists(self._eval_data_path):
            raise FileNotFoundError(f"Evaluation dataset not found: {self._eval_data_path}")

        with open(self._eval_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if split == "all":
            return data
        return [item for item in data if item.get("split") == split]

    @staticmethod
    def _check_answer_accuracy(answer: str, ground_truth: Dict[str, Any]) -> bool:
        """Check if generated answer contains key factual concepts from verified excerpt."""
        excerpt = ground_truth.get("verified_excerpt")
        if not excerpt:
            return True

        # Extract meaningful terms (length >= 4) from ground-truth excerpt
        terms = [t.lower() for t in re.findall(r"\b[A-Za-z0-9]{4,}\b", excerpt)]
        if not terms:
            return excerpt.lower() in answer.lower()

        answer_lower = answer.lower()
        matched = [t for t in terms if t in answer_lower]
        # At least 2 terms or 30% of terms present
        threshold = min(2, len(terms))
        return len(matched) >= threshold

    def evaluate(
        self,
        split: str = "evaluation",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> GroundingMetrics:
        """Run formal grounding evaluation across the chosen dataset split."""
        items = self._load_data(split)
        if not items:
            raise ValueError(f"No evaluation queries found for split: '{split}'")

        positive_items = [it for it in items if not it.get("is_negative", False)]
        negative_items = [it for it in items if it.get("is_negative", False)]

        positive_correct = 0
        evidence_supported_count = 0
        positive_abstention_errors = 0

        true_negative_abstentions = 0
        false_answers = 0
        total_abstentions_triggered = 0

        total_citations = 0
        valid_citations = 0
        supported_citations = 0
        unsupported_answers = 0

        generation_latencies: List[float] = []
        total_latencies: List[float] = []
        tokens_per_secs: List[float] = []
        query_evals: List[Dict[str, Any]] = []

        for item in items:
            q_id = item.get("query_id", "UNKNOWN")
            query_str = item["query"]
            is_neg = item.get("is_negative", False)

            ans: GroundedAnswer = self._generator.generate_answer(
                query=query_str,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            tel = ans.telemetry
            generation_latencies.append(tel.get("generation_latency_ms", 0.0))
            total_latencies.append(tel.get("total_latency_ms", 0.0))
            if tel.get("tokens_per_sec", 0.0) > 0.0:
                tokens_per_secs.append(tel.get("tokens_per_sec", 0.0))

            if ans.is_abstained:
                total_abstentions_triggered += 1

            if ans.grounding_status == "unsupported":
                unsupported_answers += 1

            for cit in ans.citations:
                total_citations += 1
                if cit.is_metadata_valid:
                    valid_citations += 1
                    if cit.is_evidence_supported:
                        supported_citations += 1

            query_record: Dict[str, Any] = {
                "query_id": q_id,
                "is_negative": is_neg,
                "is_abstained": ans.is_abstained,
                "abstention_type": ans.abstention_type,
                "grounding_status": ans.grounding_status,
                "citations_count": len(ans.citations),
                "total_latency_ms": tel.get("total_latency_ms", 0.0),
            }

            if is_neg:
                if ans.is_abstained:
                    true_negative_abstentions += 1
                    query_record["status"] = "CORRECT_ABSTENTION"
                else:
                    false_answers += 1
                    query_record["status"] = "FALSE_ANSWER_ERROR"
            else:
                if ans.is_abstained:
                    positive_abstention_errors += 1
                    query_record["status"] = "ERRONEOUS_ABSTENTION"
                else:
                    is_correct = self._check_answer_accuracy(ans.answer, item)
                    if is_correct:
                        positive_correct += 1
                    if ans.grounding_status in ("verified", "partially_supported"):
                        evidence_supported_count += 1
                    query_record["status"] = "CORRECT_ANSWER" if is_correct else "INCORRECT_ANSWER"

            query_evals.append(query_record)

        total_pos = len(positive_items)
        total_neg = len(negative_items)
        total_q = len(items)

        ans_correctness = (positive_correct / total_pos) if total_pos > 0 else 0.0
        ev_support_rate = (evidence_supported_count / total_pos) if total_pos > 0 else 0.0
        cit_validity_rate = (valid_citations / total_citations) if total_citations > 0 else 1.0
        cit_support_acc = (supported_citations / valid_citations) if valid_citations > 0 else 1.0

        neg_abstention_rate = (true_negative_abstentions / total_neg) if total_neg > 0 else 1.0
        false_ans_rate = (false_answers / total_neg) if total_neg > 0 else 0.0
        abstention_err_rate = (positive_abstention_errors / total_pos) if total_pos > 0 else 0.0

        abstention_prec = (
            (true_negative_abstentions / total_abstentions_triggered)
            if total_abstentions_triggered > 0
            else 1.0
        )
        abstention_rec = neg_abstention_rate
        unsupported_rate = (unsupported_answers / total_q) if total_q > 0 else 0.0

        mean_gen_lat = (sum(generation_latencies) / len(generation_latencies)) if generation_latencies else 0.0
        mean_tot_lat = (sum(total_latencies) / len(total_latencies)) if total_latencies else 0.0
        mean_tps = (sum(tokens_per_secs) / len(tokens_per_secs)) if tokens_per_secs else 0.0

        return GroundingMetrics(
            total_queries=total_q,
            positive_queries=total_pos,
            negative_queries=total_neg,
            answer_correctness=ans_correctness,
            evidence_support_rate=ev_support_rate,
            citation_validity_rate=cit_validity_rate,
            citation_support_accuracy=cit_support_acc,
            negative_abstention_rate=neg_abstention_rate,
            abstention_precision=abstention_prec,
            abstention_recall=abstention_rec,
            unsupported_answer_rate=unsupported_rate,
            false_answer_rate=false_ans_rate,
            abstention_errors=abstention_err_rate,
            mean_generation_latency_ms=mean_gen_lat,
            mean_total_latency_ms=mean_tot_lat,
            mean_tokens_per_sec=mean_tps,
            query_evaluations=query_evals,
        )
