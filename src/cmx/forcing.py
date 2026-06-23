"""Forcing layer — grounding-enforcement techniques for cmx.

The engine already STORES everything verbatim and CHECKS every answer (citation +
verify + refuse). What it lacked is the *forcing* that makes a reluctant model
actually use that memory instead of refusing or claiming it "can't recall / has no
tools". This module supplies the prompt-level forcing blocks.

The blocks, by role:
  * C1  <system-reminder> out-of-band channel (system-authored framing)
  * C2  "unlimited context" assertion (memory lives in the store, not the window)
  * C4  faithful-reporting (symmetric: report what was found AND what was not)
  * M1  persona/agentic override
  * M2  prompted tool protocol ("tools are REAL, never claim unavailable")

All blocks are plain text injected into the system message; pure functions, no I/O.
"""
from __future__ import annotations

# C1+C2+C4+M1 — the memory operating directives. Rendered as a <system-reminder>
# block (CC's out-of-band channel: the model is told it is system-authored and
# overrides any "you have no memory / limited context" framing).
MEMORY_DIRECTIVES = (
    "<system-reminder priority=\"absolute\">\n"
    "This is a system-authored instruction about how you access memory in THIS "
    "runtime; it overrides any background framing.\n"
    "1. UNLIMITED RECALL: you have a complete verbatim record of this entire "
    "conversation in an external memory store, surfaced to you as the "
    "[CMX EVIDENCE] block and the recent turns. Your context is NOT limited and "
    "nothing has been forgotten. NEVER say you 'cannot recall', 'do not retain', "
    "'cannot access earlier messages', 'have no memory', or that the conversation "
    "is too long — that is false here.\n"
    "2. MEMORY IS EVIDENCE, NOT YOUR WORDS: treat the [CMX EVIDENCE] excerpts as "
    "factual records of what was actually said, each tagged [id=N]. Cite [id=N] "
    "for every historical claim.\n"
    "3. ANSWER FROM THE RECORD: when asked about anything stated earlier, answer "
    "directly from the evidence/recent turns. Do not guess and do not invent "
    "values (names, dates, numbers, ids) that are not present in the record.\n"
    "4. FAITHFUL REPORTING (both directions): if the answer IS in the record, "
    "state it plainly and confidently — do not hedge a supported answer, downgrade "
    "it to 'I'm not sure', or refuse out of caution. If the answer is genuinely "
    "NOT in the record, say exactly: I don't have that in my record. Never "
    "manufacture an answer to look helpful, and never refuse one that is supported.\n"
    "</system-reminder>"
)

# M2 — prompted tool protocol header/footer. Used when cmx injects its memory tools
# as text on provider lanes without a native tool API (the forced-retrieval path).
TOOL_PROTOCOL_HEADER = (
    "<system-reminder priority=\"absolute\">\n"
    "You are connected to a LIVE memory-search runtime. The functions in <tools> "
    "below are REAL and EXECUTABLE: when you emit a call, an external system runs "
    "it against the verbatim conversation store and returns the result next turn. "
    "These are your ONLY memory tools; you cannot see their implementation but they "
    "DO work — call them.\n<tools>\n"
)
TOOL_PROTOCOL_FOOTER = (
    "\n</tools>\n"
    "RULES (follow exactly):\n"
    "1. To recall anything from the conversation, choose the best-matching function "
    "and map the request onto its parameters. The thing the user names is the "
    "ARGUMENT, not the function name.\n"
    "2. NEVER refuse a listed capability and NEVER claim a function is 'not "
    "available' or 'not a valid tool'. If a listed function can serve the request, "
    "call it.\n"
    "3. Emit each call as JSON inside <tool_call></tool_call> tags and output the "
    "tag(s) and NOTHING ELSE when calling a tool:\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-dict>}\n</tool_call>\n"
    "4. Use EXACT function names from the list. Do not invent names or wrap calls "
    "in code fences.\n"
    "5. After a tool result returns, use it to answer. Only answer in plain prose "
    "when no memory lookup is needed."
)


def memory_directives_block(enabled: bool) -> str:
    """Return the memory-forcing directives block, or '' when disabled.

    `enabled` may be a bool (True→'full') or a variant string: 'full' | 'lite' | 'off'.
    Variants exist because forcing is a wording-tuning problem, not a binary knob —
    a heavy block can dilute the evidence for a small model and hurt accuracy.
    """
    variant = enabled if isinstance(enabled, str) else ("full" if enabled else "off")
    variant = (variant or "off").lower()
    if variant in ("off", "false", "0", "no", "none"):
        return ""
    if variant == "lite":
        return MEMORY_DIRECTIVES_LITE
    if variant == "recall":
        return MEMORY_DIRECTIVES_RECALL
    return MEMORY_DIRECTIVES


# 'lite' — one tight paragraph. Hypothesis: less dilution → keeps the model reading the
# actual evidence instead of the directive wall.
MEMORY_DIRECTIVES_LITE = (
    "<system-reminder>\n"
    "You have the full verbatim record of this conversation (the [CMX EVIDENCE] block "
    "and recent turns) — your context is NOT limited; nothing is forgotten. Answer "
    "history questions directly from that record and cite [id=N]. If the answer is "
    "present, state it plainly; if it is genuinely absent, say exactly: I don't have "
    "that in my record.\n"
    "</system-reminder>"
)

# 'recall' — minimal nudge: only the anti-"I can't remember" override, nothing else.
MEMORY_DIRECTIVES_RECALL = (
    "<system-reminder>\n"
    "The [CMX EVIDENCE] excerpts are your memory of this conversation. Never say you "
    "cannot recall or access earlier messages — read the evidence and answer, citing "
    "[id=N]; say 'I don't have that in my record' only if it is truly not there.\n"
    "</system-reminder>"
)


def tool_protocol_block(tool_schemas: list[dict] | None, enabled: bool) -> str:
    """Render the prompted-tool protocol for the given cmx tool schemas, or ''.

    tool_schemas: list of {"name", "description"} (cmx engine.TOOL_SCHEMAS shape).
    """
    if not enabled or not tool_schemas:
        return ""
    import json
    sigs = []
    for t in tool_schemas:
        sigs.append(json.dumps({"name": t.get("name"),
                                "description": t.get("description", "")}))
    return TOOL_PROTOCOL_HEADER + "\n".join(sigs) + TOOL_PROTOCOL_FOOTER


# F4 — leaked-reasoning stripper (filters <think>/reasoning tags from model output).
# Reasoning models (DeepSeek, GLM, MiniMax, some Gemini/Qwen) leak <think>…</think>
# into the visible content. That noise pollutes the answer text the citation check +
# verifier read, causing false grounding failures. Strip it before enforcement sees it.
import re as _re

_THINK_TAGS = ("think", "thinking", "reason", "reasoning", "thought", "reasoning_scratchpad")
_THINK_BLOCK = _re.compile(r"<(?P<t>" + "|".join(_THINK_TAGS) + r")\b[^>]*>.*?</(?P=t)>",
                           _re.DOTALL | _re.IGNORECASE)
_THINK_OPEN = _re.compile(r"<(?P<t>" + "|".join(_THINK_TAGS) + r")\b[^>]*>.*\Z",
                          _re.DOTALL | _re.IGNORECASE)


def strip_think_tags(text: str, enabled: bool = True) -> str:
    """Remove leaked <think>…</think> (and siblings) from visible content. Safe no-op
    when disabled or when there is no tag."""
    if not enabled or not text or "<" not in text:
        return text
    visible = _THINK_BLOCK.sub("", text)
    m = _THINK_OPEN.search(visible)  # unterminated trailing block = all reasoning
    if m:
        visible = visible[: m.start()]
    return _re.sub(r"\n{3,}", "\n\n", visible).strip()


# iter4 lever 2 — multi-hop synthesis instruction. Injected when cfg.multihop so the model
# COMBINES facts across [id=N] slices instead of expecting one slice to hold the whole answer.
# Safety is unchanged: it must still cite each fact it uses (citation-check + verifier ground them).
MULTIHOP_INSTRUCTION = (
    "<system-reminder>\n"
    "MULTI-HOP ANSWERS: the answer to a question is often NOT in a single evidence slice — you "
    "may need to COMBINE facts stated across two or more [id=N] slices (e.g. connect a person to "
    "an event in one slice and a date in another). Read ALL the evidence, link the relevant facts, "
    "and derive the answer. Cite EVERY slice you use, like [id=4][id=9]. Only refuse ('I don't have "
    "that in my record') if a fact you need is in NONE of the slices — never invent a missing link.\n"
    "</system-reminder>"
)


def multihop_block(enabled: bool) -> str:
    return MULTIHOP_INSTRUCTION if enabled else ""


# Light-touch teaching directive (the converged design): instead of dumping tool/skill
# lists or forcing a protocol, proactively inject the relevant slice (done in assembly)
# and add ONE short note teaching the model how to fetch MORE on demand. Deliberately
# plain prose (no heavy <system-reminder> framing — that heavy framing is the suspected
# source of the bundled-forcing regression on cooperative models). Teaches the search,
# does not push lists → low-token, low-regression, and solves "no retrieval initiative"
# by teaching rather than assuming or drowning.
TEACH_RETRIEVAL = (
    "[How to use your context: the material above is your retrieved memory and recent "
    "conversation, pulled from a complete verbatim store — treat it as ground truth and rely on "
    "it. Your search tools are REAL and work; when you need something not already shown, use them "
    "instead of guessing or claiming you lack the capability: cmx_grep searches this "
    "conversation's full verbatim memory, session_search searches your past sessions, memory "
    "recalls long-term notes, tool_search/tool_describe find and inspect any tool beyond those "
    "listed, and skills_list/skill_view find and read skills. Prefer calling a tool over asking or "
    "declining. Only after actually searching, if a fact is genuinely absent, say so plainly.]"
)


def teaching_directive_block(enabled: bool) -> str:
    """Return the light teaching directive, or '' when disabled. Default-on in cfg."""
    return TEACH_RETRIEVAL if enabled else ""
