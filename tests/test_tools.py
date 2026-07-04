"""Tests for the GitHub LangChain tools.

HTTP is mocked with respx. The "LLM calls the tool" DoD item is verified with
a *fake* chat model (no real API key): the fake emits a tool_call for
`fetch_pr_diff`, and we prove that tool_call actually drives the tool.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.tools.github_tools import (
    GITHUB_TOOLS,
    fetch_pr_diff,
    fetch_pr_files,
    post_pr_review,
)

BASE = "https://api.github.com"


# --------------------------------------------------------------------------- #
# Standalone invocation (structured envelope)
# --------------------------------------------------------------------------- #
@respx.mock
async def test_fetch_pr_diff_ok_envelope():
    respx.get(f"{BASE}/repos/octo/hello/pulls/5").mock(
        return_value=httpx.Response(200, text="diff --git a/x b/x\n+hi")
    )
    result = await fetch_pr_diff.ainvoke(
        {"owner": "octo", "repo": "hello", "pr_number": 5}
    )
    assert result["ok"] is True
    assert result["error"] is None
    assert "diff --git" in result["data"]


@respx.mock
async def test_fetch_pr_files_ok_envelope():
    respx.get(f"{BASE}/repos/octo/hello/pulls/5/files").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "filename": "app.py",
                    "status": "modified",
                    "additions": 3,
                    "deletions": 1,
                    "changes": 4,
                    "patch": "@@ -1 +1 @@\n-x\n+y",
                }
            ],
        )
    )
    result = await fetch_pr_files.ainvoke(
        {"owner": "octo", "repo": "hello", "pr_number": 5}
    )
    assert result["ok"] is True
    assert result["data"][0]["filename"] == "app.py"


@respx.mock
async def test_post_pr_review_ok_envelope():
    respx.post(f"{BASE}/repos/octo/hello/issues/5/comments").mock(
        return_value=httpx.Response(
            201,
            json={"id": 7, "html_url": "https://gh/c/7", "body": "LGTM"},
        )
    )
    result = await post_pr_review.ainvoke(
        {"owner": "octo", "repo": "hello", "pr_number": 5, "body": "LGTM"}
    )
    assert result["ok"] is True
    assert result["data"]["id"] == 7


@respx.mock
async def test_tool_returns_error_envelope_not_exception():
    respx.get(f"{BASE}/repos/octo/hello/pulls/999").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    result = await fetch_pr_diff.ainvoke(
        {"owner": "octo", "repo": "hello", "pr_number": 999}
    )
    assert result["ok"] is False
    assert result["data"] is None
    assert result["error"] == "Not Found"


# --------------------------------------------------------------------------- #
# Tool metadata (the LLM reads these)
# --------------------------------------------------------------------------- #
def test_tools_have_descriptions_and_schemas():
    for t in GITHUB_TOOLS:
        assert t.description and len(t.description) > 20
        assert t.args_schema is not None
    names = {t.name for t in GITHUB_TOOLS}
    assert names == {"fetch_pr_diff", "fetch_pr_files", "post_pr_review"}


# --------------------------------------------------------------------------- #
# Mocked tool-calling (no real API key)
# --------------------------------------------------------------------------- #
def test_tools_convert_to_llm_tool_schema():
    """Each tool converts to a valid LLM tool schema — what bind_tools does."""
    for t in GITHUB_TOOLS:
        schema = convert_to_openai_tool(t)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == t.name
        assert fn["description"]
        # Args are exposed to the model as JSON-schema properties.
        assert "owner" in fn["parameters"]["properties"]
    # The post tool additionally exposes the review body.
    post_schema = convert_to_openai_tool(post_pr_review)
    assert "body" in post_schema["function"]["parameters"]["properties"]


@respx.mock
async def test_model_toolcall_drives_the_tool():
    # The fake model "decides" to call fetch_pr_diff (stands in for a real LLM
    # asked: "fetch the diff for PR 5 in octo/hello").
    planned = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "fetch_pr_diff",
                "args": {"owner": "octo", "repo": "hello", "pr_number": 5},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    model = FakeMessagesListChatModel(responses=[planned])
    ai = model.invoke("fetch the diff for PR 5 in octo/hello")
    assert ai.tool_calls, "model should have emitted a tool call"
    call = ai.tool_calls[0]
    assert call["name"] == "fetch_pr_diff"

    # Route the tool_call to the actual tool and execute it.
    respx.get(f"{BASE}/repos/octo/hello/pulls/5").mock(
        return_value=httpx.Response(200, text="diff --git a/z b/z\n+ok")
    )
    tool_map = {t.name: t for t in GITHUB_TOOLS}
    tool_msg = await tool_map[call["name"]].ainvoke(call)  # returns a ToolMessage

    assert "diff --git" in tool_msg.content
    assert '"ok": true' in tool_msg.content.lower()
