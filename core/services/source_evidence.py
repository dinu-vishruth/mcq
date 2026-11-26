"""Server-signed acceptance receipts bind displayed explanations to checked MCQs."""
import hashlib
import hmac
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar

import config

_FIELDS = ("question", "options", "answer_text", "explanation", "source_hint")
_locations = ContextVar("source_locations", default=())


@contextmanager
def source_locations(hits):
    token = _locations.set(tuple(hits))
    try:
        yield
    finally:
        _locations.reset(token)


def _signature(question, context_hash, provenance=None):
    payload = {key: question.get(key) for key in _FIELDS}
    payload["context_sha256"] = context_hash
    if provenance is not None:
        payload["provenance"] = provenance
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hmac.new(str(config.SECRET_KEY).encode(), serialized.encode(), hashlib.sha256).hexdigest()


def issue(question, context):
    context_hash = hashlib.sha256(context.encode()).hexdigest()
    from core.models.mcq_contract import normalized_text
    for hit in _locations.get():
        if normalized_text(question["source_hint"]) in normalized_text(hit.content):
            provenance = {"document_id": hit.document_id, "chunk_index": hit.chunk_index,
                          "chunk_sha256": hashlib.sha256(hit.content.encode()).hexdigest()}
            return {"quote": question["source_hint"], "context_sha256": context_hash,
                    "contract": "grounded-v3", "provenance": provenance,
                    "receipt": _signature(question, context_hash, provenance)}
    return {"quote": question["source_hint"], "context_sha256": context_hash,
            "contract": "grounded-v2", "receipt": _signature(question, context_hash)}


def authentic(question):
    if not isinstance(question, dict):
        return False
    evidence = question.get("source_evidence")
    if not isinstance(evidence, dict) or evidence.get("contract") not in ("grounded-v2", "grounded-v3"):
        return False
    digest, receipt = evidence.get("context_sha256"), evidence.get("receipt")
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (digest, receipt)):
        return False
    try:
        provenance = evidence.get("provenance") if evidence["contract"] == "grounded-v3" else None
        if evidence["contract"] == "grounded-v3" and not isinstance(provenance, dict):
            return False
        return (question.get("verification_status") == "verified"
                and question.get("safety_status") == "verified"
                and evidence.get("quote") == question.get("source_hint")
                and hmac.compare_digest(receipt, _signature(question, digest, provenance)))
    except (TypeError, ValueError):
        return False
