"""Server-signed acceptance receipts bind displayed explanations to checked MCQs."""
import hashlib
import hmac
import json
import re

import config

_FIELDS = ("question", "options", "answer_text", "explanation", "source_hint")


def _signature(question, context_hash):
    payload = {key: question.get(key) for key in _FIELDS}
    payload["context_sha256"] = context_hash
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hmac.new(str(config.SECRET_KEY).encode(), serialized.encode(), hashlib.sha256).hexdigest()


def issue(question, context):
    context_hash = hashlib.sha256(context.encode()).hexdigest()
    return {"quote": question["source_hint"], "context_sha256": context_hash,
            "contract": "grounded-v2", "receipt": _signature(question, context_hash)}


def authentic(question):
    if not isinstance(question, dict):
        return False
    evidence = question.get("source_evidence")
    if not isinstance(evidence, dict) or evidence.get("contract") != "grounded-v2":
        return False
    digest, receipt = evidence.get("context_sha256"), evidence.get("receipt")
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (digest, receipt)):
        return False
    try:
        return (question.get("verification_status") == "verified"
                and question.get("safety_status") == "verified"
                and evidence.get("quote") == question.get("source_hint")
                and hmac.compare_digest(receipt, _signature(question, digest)))
    except (TypeError, ValueError):
        return False
