"""Model-agnostic tool-call normalization layer.

Some hosted proxies front many model families (OpenAI, Claude, Gemini, Grok,
DeepSeek, Llama, Mistral, ...) behind a single endpoint with NO native tool
API, so the only way to get tool use is to inject a prompted tool protocol and
parse tool calls back out of the model's TEXT.

Problem: every family emits a slightly different "I want to call a tool" syntax,
and they drift over time. Hard-coding one regex per surprise is whack-a-mole.

This module is the durable answer: a REGISTRY of small "dialect" extractors,
tried in priority order, each returning best-effort RawToolCall objects. The
public entrypoint normalizes whatever the model produced into OpenAI-format
tool_calls[], optionally validating names/arguments against the declared tool
schema so we can keep recall high without false positives on ordinary prose.

Adding support for a new model's quirk = add ONE dialect function + register it.
No change to callers, no change to the chat route.

Dialects covered (real output observed across prompted-tool providers):
  * nous_xml        <tool_call>{json}</tool_call>            (OpenAI/Claude/Grok/DeepSeek/Gemini via our prompt)
  * anthropic_xml   <function_calls><invoke name=..><parameter ..>  (Claude native style)
  * gemini_code     <tool_code> ... </tool_code>             (Gemini)
  * fenced_json     ```tool_call / ```json {json} ```        (any model that fences)
  * python_call     fn({...})  /  fn(city="Paris")  /  print(fn.run({...}))   (Gemini/code models)

Design notes:
  * python_call uses ast.parse, NOT regex+json — it must handle Python kwargs
    (fn(city="Paris", units='c')) and single quotes that json.loads rejects.
  * The bare python_call dialect only fires when tool_names is known, so prose
    like "use the index(...) method" is never mistaken for a tool call.
  * Each dialect reports the (start, end) spans it consumed so the caller can
    strip tool-call syntax out of the user-visible content.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RawToolCall:
    name: str
    arguments: dict[str, Any]
    span: tuple[int, int] | None = None  # (start, end) in source text, if known


# A dialect takes the model text (+ known tool names) and returns RawToolCalls.
Dialect = Callable[[str, "set[str] | None"], list[RawToolCall]]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _loads_loose(raw: str) -> Any | None:
    """json.loads tolerant of single quotes / trailing commas / py literals."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        # Python literal: handles single quotes, True/False/None, tuples.
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        return None


def _as_args_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj
    return None


# --------------------------------------------------------------------------- #
# dialects
# --------------------------------------------------------------------------- #

_NOUS_XML_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _d_nous_xml(text: str, _names: set[str] | None) -> list[RawToolCall]:
    out: list[RawToolCall] = []
    for m in _NOUS_XML_RE.finditer(text):
        obj = _loads_loose(m.group(1))
        if isinstance(obj, dict) and obj.get("name"):
            args = _as_args_dict(obj.get("arguments", {})) or {}
            out.append(RawToolCall(obj["name"], args, m.span()))
    return out


_FENCED_RE = re.compile(r"```(?:tool_call|tool_code|json|python)?\s*(\{.*?\})\s*```", re.DOTALL)


def _d_fenced_json(text: str, _names: set[str] | None) -> list[RawToolCall]:
    out: list[RawToolCall] = []
    for m in _FENCED_RE.finditer(text):
        obj = _loads_loose(m.group(1))
        if isinstance(obj, dict) and obj.get("name") and "arguments" in obj:
            args = _as_args_dict(obj.get("arguments", {})) or {}
            out.append(RawToolCall(obj["name"], args, m.span()))
    return out


_GEMINI_CODE_RE = re.compile(r"<tool_code>\s*(.*?)\s*</tool_code>", re.DOTALL)


def _d_gemini_code(text: str, names: set[str] | None) -> list[RawToolCall]:
    out: list[RawToolCall] = []
    for m in _GEMINI_CODE_RE.finditer(text):
        inner = m.group(1)
        # JSON form first
        obj = _loads_loose(inner)
        if isinstance(obj, dict) and obj.get("name"):
            args = _as_args_dict(obj.get("arguments", {})) or {}
            out.append(RawToolCall(obj["name"], args, m.span()))
            continue
        # else python-call form inside the block (names not required here —
        # being inside <tool_code> is signal enough)
        for rc in _extract_python_calls(inner, names=None, require_names=False):
            out.append(RawToolCall(rc.name, rc.arguments, m.span()))
    return out


_ANTHROPIC_RE = re.compile(
    r"<function_calls>\s*(.*?)\s*</function_calls>", re.DOTALL
)
_ANTHROPIC_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL
)
_ANTHROPIC_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', re.DOTALL
)


def _d_anthropic_xml(text: str, _names: set[str] | None) -> list[RawToolCall]:
    out: list[RawToolCall] = []
    for block in _ANTHROPIC_RE.finditer(text):
        for inv in _ANTHROPIC_INVOKE_RE.finditer(block.group(1)):
            name = inv.group(1)
            args: dict[str, Any] = {}
            for p in _ANTHROPIC_PARAM_RE.finditer(inv.group(2)):
                key, val = p.group(1), p.group(2).strip()
                # try to type-coerce JSON-ish values, else keep string
                coerced = _loads_loose(val)
                args[key] = coerced if coerced is not None else val
            out.append(RawToolCall(name, args, block.span()))
    return out


# fn({...})  |  fn(city="Paris")  |  print(fn.run({...}))  |  fn.invoke(...)
_PY_CALL_RE = re.compile(
    r"(?:print\(\s*)?"
    r"([A-Za-z_][A-Za-z0-9_]*)"          # 1: object-or-func name
    r"(?:\s*\.\s*(?:run|invoke|call|execute))?"  # optional .run/.invoke/...
    r"\s*\((.*?)\)",                       # 2: arg blob (non-greedy)
    re.DOTALL,
)


def _extract_python_calls(
    text: str, names: set[str] | None, require_names: bool
) -> list[RawToolCall]:
    """Parse fn(...) / fn(k=v) / fn({...}) via ast, robust to py-literal args."""
    out: list[RawToolCall] = []
    for m in _PY_CALL_RE.finditer(text):
        name = m.group(1)
        if require_names and (not names or name not in names):
            continue
        arg_blob = m.group(2).strip()
        args = _parse_call_args(name, arg_blob)
        if args is None:
            continue
        out.append(RawToolCall(name, args, m.span()))
    return out


def _parse_call_args(name: str, arg_blob: str) -> dict[str, Any] | None:
    """Return an args dict from a Python call's argument text.

    Handles: fn({"city":"Paris"}) | fn(city="Paris", units='c') | fn() .
    Uses ast so single quotes and python literals work where json.loads fails.
    """
    if arg_blob == "":
        return {}
    # Single positional dict?  fn({...})
    obj = _loads_loose(arg_blob)
    if isinstance(obj, dict):
        return obj
    # General Python call: parse `name(<blob>)` as an AST Call.
    try:
        expr = ast.parse(f"_f({arg_blob})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Call):
        return None
    args: dict[str, Any] = {}
    # keyword args -> dict
    for kw in expr.keywords:
        if kw.arg is None:
            # **kwargs splat: try to fold a literal dict
            try:
                val = ast.literal_eval(kw.value)
                if isinstance(val, dict):
                    args.update(val)
            except (ValueError, SyntaxError):
                pass
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            # non-literal (a name/expr) — store source text
            args[kw.arg] = ast.get_source_segment(arg_blob, kw.value) or None
    # a single positional literal dict?  fn({...}) already handled above;
    # other positionals are ignored (no param names to bind them to).
    if not args and expr.args:
        for a in expr.args:
            try:
                val = ast.literal_eval(a)
                if isinstance(val, dict):
                    args.update(val)
            except (ValueError, SyntaxError):
                continue
    return args


def _d_python_call(text: str, names: set[str] | None) -> list[RawToolCall]:
    # Only fire on bare text when we know the tool names (avoid prose FPs).
    return _extract_python_calls(text, names=names, require_names=True)


# --------------------------------------------------------------------------- #
# registry (priority order: most explicit / least ambiguous first)
# --------------------------------------------------------------------------- #

@dataclass
class _Registered:
    name: str
    fn: Dialect
    # model substrings this dialect is the NATIVE format for (tried first)
    prefers: tuple[str, ...] = field(default_factory=tuple)


_DIALECTS: list[_Registered] = [
    _Registered("nous_xml", _d_nous_xml),
    _Registered("anthropic_xml", _d_anthropic_xml, prefers=("claude", "anthropic")),
    _Registered("gemini_code", _d_gemini_code, prefers=("gemini",)),
    _Registered("fenced_json", _d_fenced_json),
    _Registered("python_call", _d_python_call, prefers=("gemini", "code")),
]


def register_dialect(
    name: str, fn: Dialect, *, prefers: tuple[str, ...] = (), priority: int | None = None
) -> None:
    """Register a new tool-call dialect. priority=0 makes it tried first."""
    reg = _Registered(name, fn, prefers)
    if priority is None:
        _DIALECTS.append(reg)
    else:
        _DIALECTS.insert(max(0, priority), reg)


def _ordered_dialects(model: str | None) -> list[_Registered]:
    if not model:
        return list(_DIALECTS)
    ml = model.lower()
    preferred = [d for d in _DIALECTS if any(p in ml for p in d.prefers)]
    rest = [d for d in _DIALECTS if d not in preferred]
    return preferred + rest


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def normalize_tool_calls(
    text: str,
    *,
    model: str | None = None,
    tool_names: set[str] | None = None,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize any model's tool-call text → OpenAI tool_calls[].

    Returns (text_without_tool_calls, openai_tool_calls). Empty list when none.

    model:        optional model id; biases dialect order toward the native one.
    tool_names:   declared tool names; enables the bare python_call dialect and
                  filters hallucinated names.
    tool_schemas: optional {name: json-schema}; reserved for argument coercion.
    """
    if not text:
        return text, []

    raws: list[RawToolCall] = []
    for d in _ordered_dialects(model):
        found = d.fn(text, tool_names)
        if found:
            raws = found
            break  # first dialect that matches wins (avoid double-counting)

    if not raws:
        return text, []

    # Filter against declared tool names when we have them.
    if tool_names:
        raws = [r for r in raws if r.name in tool_names] or raws

    tool_calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for r in raws:
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": r.name, "arguments": json.dumps(r.arguments)},
        })
        if r.span:
            spans.append(r.span)

    cleaned = text
    for start, end in sorted(spans, key=lambda s: s[0], reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned.strip(), tool_calls
