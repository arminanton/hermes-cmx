"""Model client interface + a scriptable mock + a Hermes auxiliary adapter.

The engine depends only on the ``ModelClient`` protocol, so the whole grounding
pipeline is unit-testable with deterministic mocks (including a *disobedient*
model that tries to guess — which the engine must still force into grounding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Completion:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelClient(Protocol):
    def complete(self, messages: list[dict], *, model: str,
                 tool_choice: Optional[str] = None,
                 tools: Optional[list] = None,
                 temperature: Optional[float] = None) -> Completion: ...


class MockModel:
    """Scriptable model for tests.

    ``script`` is a list of callables ``(messages, tool_choice) -> Completion`` or
    plain ``Completion``/str, consumed one per ``complete`` call. When exhausted,
    repeats the last entry. Lets a test simulate obedient vs guessing behavior and
    forced-tool rounds.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []
        self._i = 0

    def complete(self, messages, *, model, tool_choice=None, tools=None, temperature=None) -> Completion:
        self.calls.append({"messages": messages, "tool_choice": tool_choice})
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if callable(item):
            item = item(messages, tool_choice)
        if isinstance(item, str):
            return Completion(content=item)
        return item


class HermesAuxClient:
    """Adapter onto Hermes' ``agent.auxiliary_client.call_llm`` (used in production).

    Kept import-lazy so the package imports without Hermes installed.
    """

    def __init__(self, task: str = "compression"):
        self.task = task

    def complete(self, messages, *, model, tool_choice=None, tools=None, temperature=None) -> Completion:
        from agent.auxiliary_client import call_llm  # type: ignore
        kwargs: dict[str, Any] = {"task": self.task, "messages": messages}
        if model:
            kwargs["model"] = model
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = call_llm(**kwargs)
        msg = resp.choices[0].message
        content = msg.content if isinstance(msg.content, str) else (str(msg.content) if msg.content else "")
        calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                import json
                calls.append(ToolCall(name=tc.function.name, args=json.loads(tc.function.arguments or "{}")))
            except Exception:
                pass
        return Completion(content=content, tool_calls=calls)
