#!/usr/bin/env python
"""Black-box endpoint tests for async agent resolution.

These tests intentionally exercise only the public FastAPI endpoint contract:
the resolver is an HTTP request-time routing abstraction, state extraction is a
pre-routing concern, and open tool-call ownership is pinned in-process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunStartedEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)

from ag_ui_adk.adk_agent import ADKAgent
from ag_ui_adk.endpoint import add_adk_fastapi_endpoint, create_adk_app


def _run_input(
    *,
    thread_id: str = "thread-1",
    run_id: str = "run-1",
    messages=None,
    state=None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        messages=(
            messages
            if messages is not None
            else [UserMessage(id="user-1", role="user", content="hello")]
        ),
        tools=[],
        context=[],
        state={} if state is None else state,
        forwarded_props={},
    )


def _agent(name: str, *, capabilities=None, events=None):
    agent = MagicMock(spec=ADKAgent)
    agent.name = name
    agent.get_capabilities.return_value = capabilities

    async def run(input_data):
        for event in events or [
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
            )
        ]:
            yield event

    agent.run = MagicMock(side_effect=run)
    return agent


def _state_agent(name: str, state: dict):
    agent = _agent(name)
    adk_agent = MagicMock()
    adk_agent.name = name
    agent._adk_agent = adk_agent
    agent._static_app_name = f"{name}_app"
    agent._static_user_id = f"{name}_user"
    agent._session_lookup_cache = {}
    agent._get_session_metadata = MagicMock(
        return_value=(f"{name}_session", f"{name}_app", f"{name}_user")
    )
    agent._session_manager = MagicMock()
    agent._session_manager.get_session_state = AsyncMock(return_value=state)
    agent._session_manager._session_service = MagicMock()
    session = MagicMock()
    session.events = []
    agent._session_manager._session_service.get_session = AsyncMock(
        return_value=session
    )
    return agent


def test_resolver_runs_after_extractor_and_can_fallback_to_default_agent():
    default_agent = _agent("default")
    selected_agent = _agent("selected")
    resolver_inputs = []

    async def extractor(request, input_data):
        return {"tenant": request.headers["x-tenant"], "from_extractor": True}

    async def resolver(request, input_data):
        resolver_inputs.append(input_data)
        if input_data.state["tenant"] == "selected":
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app,
        default_agent,
        path="/agent",
        extract_state_from_request=extractor,
        agent_resolver=resolver,
    )
    client = TestClient(app)

    selected_response = client.post(
        "/agent",
        json=_run_input(state={"client_state": "preserved"}).model_dump(),
        headers={"x-tenant": "selected"},
    )
    fallback_response = client.post(
        "/agent",
        json=_run_input(run_id="run-2").model_dump(),
        headers={"x-tenant": "unknown"},
    )

    assert selected_response.status_code == 200
    assert fallback_response.status_code == 200
    assert selected_agent.run.call_count == 1
    assert default_agent.run.call_count == 1
    assert resolver_inputs[0].state == {
        "client_state": "preserved",
        "tenant": "selected",
        "from_extractor": True,
    }


def test_resolver_can_route_by_request_headers_and_query_params():
    default_agent = _agent("default")
    selected_agent = _agent("selected")

    async def resolver(request, input_data):
        if (
            request.headers.get("x-route-agent") == "selected"
            and request.query_params.get("region") == "west"
        ):
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(app, default_agent, path="/agent", agent_resolver=resolver)
    client = TestClient(app)

    response = client.post(
        "/agent?region=west",
        json=_run_input().model_dump(),
        headers={"x-route-agent": "selected"},
    )

    assert response.status_code == 200
    selected_agent.run.assert_called_once()
    default_agent.run.assert_not_called()


def test_create_adk_app_forwards_agent_resolver_functionally():
    default_agent = _agent("default")
    selected_agent = _agent("selected")

    async def resolver(request, input_data):
        return selected_agent if input_data.state.get("agent") == "selected" else None

    app = create_adk_app(default_agent, path="/agent", agent_resolver=resolver)
    client = TestClient(app)

    response = client.post(
        "/agent", json=_run_input(state={"agent": "selected"}).model_dump()
    )

    assert response.status_code == 200
    selected_agent.run.assert_called_once()
    default_agent.run.assert_not_called()


def test_capabilities_uses_resolver_after_extractor_and_defaults_on_none():
    default_agent = _agent("default", capabilities={"identity": {"name": "default"}})
    selected_agent = _agent("selected", capabilities={"identity": {"name": "selected"}})
    resolver_inputs = []

    async def extractor(request, input_data):
        if "x-capability-agent" in request.headers:
            return {"capability_agent": request.headers["x-capability-agent"]}
        return {}

    async def resolver(request, input_data):
        resolver_inputs.append(input_data)
        if input_data.state.get("capability_agent") == "selected":
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app,
        default_agent,
        path="/agent",
        extract_state_from_request=extractor,
        agent_resolver=resolver,
    )
    client = TestClient(app)

    selected_response = client.get(
        "/agent/capabilities", headers={"x-capability-agent": "selected"}
    )
    fallback_response = client.get("/agent/capabilities")

    assert selected_response.status_code == 200
    assert selected_response.json()["identity"]["name"] == "selected"
    assert fallback_response.status_code == 200
    assert fallback_response.json()["identity"]["name"] == "default"
    assert resolver_inputs[0].state == {"capability_agent": "selected"}
    assert resolver_inputs[0].messages == []


def test_agents_state_uses_resolved_agent_after_extractor_merge():
    default_agent = _state_agent("default", {"source": "default"})
    selected_agent = _state_agent("selected", {"source": "selected"})
    resolver_inputs = []

    async def extractor(request, input_data):
        return {"state_agent": request.headers["x-state-agent"]}

    async def resolver(request, input_data):
        resolver_inputs.append(input_data)
        if input_data.state["state_agent"] == "selected":
            return selected_agent
        return None

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app,
        default_agent,
        path="/",
        extract_state_from_request=extractor,
        agent_resolver=resolver,
    )
    client = TestClient(app)

    response = client.post(
        "/agents/state",
        json={"threadId": "thread-state"},
        headers={"x-state-agent": "selected"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == {"source": "selected"}
    assert resolver_inputs[0].thread_id == "thread-state"
    assert resolver_inputs[0].state == {"state_agent": "selected"}
    selected_agent._session_manager.get_session_state.assert_awaited_once()
    default_agent._session_manager.get_session_state.assert_not_awaited()


def test_open_tool_call_routes_follow_up_tool_message_to_pinned_agent():
    pinned_agent = _agent(
        "pinned",
        events=[
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id="tool-call-1",
                tool_call_name="needs_client_result",
            )
        ],
    )
    other_agent = _agent("other")
    resolver = AsyncMock(
        side_effect=lambda request, input_data: (
            pinned_agent if input_data.state.get("agent") == "pinned" else other_agent
        )
    )

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app, _agent("default"), path="/agent", agent_resolver=resolver
    )
    client = TestClient(app)

    first_response = client.post(
        "/agent",
        json=_run_input(state={"agent": "pinned"}).model_dump(),
    )
    second_response = client.post(
        "/agent",
        json=_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="tool-message-1",
                    role="tool",
                    tool_call_id="tool-call-1",
                    content='{"ok": true}',
                )
            ],
            state={"agent": "other"},
        ).model_dump(),
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert pinned_agent.run.call_count == 2
    other_agent.run.assert_not_called()
    assert resolver.await_count == 1


def test_unpinned_tool_message_returns_run_error_without_resolver_or_agent():
    resolver = AsyncMock(return_value=_agent("resolved"))
    default_agent = _agent("default")

    app = FastAPI()
    add_adk_fastapi_endpoint(app, default_agent, path="/agent", agent_resolver=resolver)
    client = TestClient(app)

    response = client.post(
        "/agent",
        json=_run_input(
            messages=[
                ToolMessage(
                    id="tool-message-1",
                    role="tool",
                    tool_call_id="missing-tool-call",
                    content='{"ok": true}',
                )
            ]
        ).model_dump(),
    )

    assert response.status_code == 200
    assert '"type":"RUN_ERROR"' in response.text
    resolver.assert_not_awaited()
    default_agent.run.assert_not_called()


def test_mixed_pinned_and_unpinned_tool_messages_return_run_error():
    pinned_agent = _agent(
        "pinned",
        events=[
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id="tool-call-1",
                tool_call_name="needs_client_result",
            )
        ],
    )
    resolver = AsyncMock(return_value=pinned_agent)

    app = FastAPI()
    add_adk_fastapi_endpoint(
        app, _agent("default"), path="/agent", agent_resolver=resolver
    )
    client = TestClient(app)

    client.post("/agent", json=_run_input().model_dump())
    resolver.reset_mock()
    pinned_agent.run.reset_mock()

    response = client.post(
        "/agent",
        json=_run_input(
            run_id="run-2",
            messages=[
                ToolMessage(
                    id="tool-message-1",
                    role="tool",
                    tool_call_id="tool-call-1",
                    content='{"ok": true}',
                ),
                ToolMessage(
                    id="tool-message-2",
                    role="tool",
                    tool_call_id="missing-tool-call",
                    content='{"ok": true}',
                ),
            ],
        ).model_dump(),
    )

    assert response.status_code == 200
    assert '"type":"RUN_ERROR"' in response.text
    resolver.assert_not_awaited()
    pinned_agent.run.assert_not_called()
