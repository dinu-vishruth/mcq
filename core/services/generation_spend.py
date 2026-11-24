"""Conservative per-generation reservations, shared across retries and recovery.

Reservations are not refunded after errors: providers may bill failed attempts.
USD enforcement requires operator-supplied rates; this is not a billing meter.
"""
import math
from contextvars import ContextVar
from dataclasses import dataclass

_ledger = ContextVar("generation_spend", default=None)


class GenerationSpendExceeded(RuntimeError):
    pass


@dataclass
class SpendLedger:
    calls: int = 0
    reserved_tokens: int = 0
    reserved_usd: float = 0.0


def begin():
    return _ledger.set(SpendLedger())


def end(token):
    _ledger.reset(token)


def snapshot():
    ledger = _ledger.get()
    return vars(ledger).copy() if ledger else None


def reserve(messages, max_tokens):
    import config
    ledger = _ledger.get()
    if ledger is None:
        raise GenerationSpendExceeded("No generation spend budget is active.")
    output = config.LLM_DEFAULT_MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
    if type(output) is not int or not 1 <= output <= config.LLM_MAX_OUTPUT_TOKENS:
        raise GenerationSpendExceeded("Invalid output token allowance.")
    # UTF-8 bytes plus message framing is deliberately more conservative than
    # chars/4. Exact tokenization and invoices still depend on the provider.
    input_tokens = 256 + sum(len(m.get("content", "").encode("utf-8")) + 64 for m in messages)
    tokens = input_tokens + output
    cap = config.GENERATION_MAX_COST_USD
    if not math.isfinite(cap) or cap < 0:
        raise GenerationSpendExceeded("Invalid generation cost cap.")
    input_rate, output_rate = config.LLM_INPUT_USD_PER_MILLION, config.LLM_OUTPUT_USD_PER_MILLION
    usd = 0.0
    if cap > 0:
        if not all(math.isfinite(v) and v > 0 for v in (cap, input_rate, output_rate)):
            raise GenerationSpendExceeded("USD budget requires positive configured provider rates.")
        usd = (input_tokens * input_rate + output * output_rate) / 1_000_000
    if (ledger.calls + 1 > config.GENERATION_MAX_LLM_CALLS
            or ledger.reserved_tokens + tokens > config.GENERATION_MAX_RESERVED_TOKENS
            or (cap > 0 and ledger.reserved_usd + usd > cap)):
        raise GenerationSpendExceeded("Generation token or spend limit reached.")
    ledger.calls += 1
    ledger.reserved_tokens += tokens
    ledger.reserved_usd += usd
    return output
