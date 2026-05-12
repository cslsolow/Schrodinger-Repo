"""Tests for SemanticMappingAgent."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.agents.mapped_agent import SemanticMappingAgent


@pytest.fixture
def mapping_dir(tmp_path):
    mapping = {"get_queryset": "fetch_dataset"}
    (tmp_path / "mapping_seed_42.json").write_text(json.dumps(mapping))
    (tmp_path / "token_mapping.json").write_text(json.dumps({}))
    return tmp_path


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.format_message.side_effect = lambda **kwargs: kwargs
    model.get_template_vars.return_value = {}
    model.serialize.return_value = {}
    model.format_observation_messages.return_value = []
    return model


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.get_template_vars.return_value = {}
    env.serialize.return_value = {}
    return env


def test_task_translated_to_virtual(mapping_dir, mock_model, mock_env):
    """run() should translate task description to virtual space."""
    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(mapping_dir, seed=42)

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    # Patch step to immediately exit
    def fake_step(self_ref=agent):
        self_ref.add_messages({"role": "exit", "extra": {"exit_status": "test", "submission": ""}})
        return []

    with patch.object(agent, 'step', side_effect=fake_step):
        agent.run("fix the get_queryset method")

    # The instance message should contain virtual name
    instance_msg = agent.messages[1]  # messages[0] is system, [1] is instance
    assert "fetch_dataset" in instance_msg.get("content", "")
    assert "get_queryset" not in instance_msg.get("content", "")


def test_run_uses_text_channel_for_identity_task_sanitization(tmp_path, mock_model, mock_env):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(
        json.dumps({"django.helperkit": "django.utilities", "django.contrib": "django.auxiliary"})
    )
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    agent = SemanticMappingAgent(
        mock_model,
        mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    def fake_step(self_ref=agent):
        self_ref.add_messages({"role": "exit", "extra": {"exit_status": "test", "submission": ""}})
        return []

    task = (
        "Django 2.1 issue in `from django.helperkit import dateformat` and "
        "`django.contrib.admin` should preserve repo identity masking."
    )

    with patch.object(agent, "step", side_effect=fake_step):
        agent.run(task)

    instance_msg = agent.messages[1]
    content = instance_msg.get("content", "")
    assert "working_repository 2.1" in content
    assert "working_repository.utilities" in content
    assert "working_repository.auxiliary.admin" in content
    assert "Django" not in content
    assert "django." not in content


def test_command_translated_to_real(mapping_dir, mock_model, mock_env):
    """execute_actions should translate command from virtual to real."""
    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(mapping_dir, seed=42)

    # Pre-populate notebook (as if to_virtual had been called)
    manager.session_notebook["fetch_dataset"] = "get_queryset"

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    # Simulate env.execute returning real output
    mock_env.execute.return_value = {
        "output": "def get_queryset(self):\n    pass",
        "returncode": 0,
        "exception_info": "",
    }

    message = {
        "content": "Let me search for fetch_dataset",
        "role": "assistant",
        "extra": {
            "actions": [{"command": "grep -r 'fetch_dataset' /testbed/", "tool_call_id": "tc1"}],
        },
    }

    mock_model.format_observation_messages.return_value = [
        {"role": "tool", "content": "output", "tool_call_id": "tc1"}
    ]

    agent.execute_actions(message)

    # The command sent to env should be translated to real
    call_args = mock_env.execute.call_args
    executed_action = call_args[0][0]
    assert "get_queryset" in executed_action["command"]
    assert "fetch_dataset" not in executed_action["command"]


def test_no_manager_passthrough(mock_model, mock_env):
    """Without mapping_manager, agent should behave like DefaultAgent."""
    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=None,
        system_template="system",
        instance_template="{{task}}",
    )

    mock_env.execute.return_value = {"output": "ok", "returncode": 0, "exception_info": ""}
    mock_model.format_observation_messages.return_value = [{"role": "tool", "content": "ok", "tool_call_id": "tc1"}]

    message = {
        "content": "test",
        "role": "assistant",
        "extra": {"actions": [{"command": "echo hello", "tool_call_id": "tc1"}]},
    }

    agent.execute_actions(message)
    call_args = mock_env.execute.call_args
    assert call_args[0][0]["command"] == "echo hello"


def test_query_preserves_assistant_thought_text_but_sanitizes_commands(tmp_path, mock_model, mock_env):
    """query() should preserve natural-language thought text while sanitizing command blocks/actions."""
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(
        json.dumps({"Count": "Tally", "run": "carry_out", "ll": "low_left"})
    )
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    mock_model.query.return_value = {
        "role": "assistant",
        "content": (
            "THOUGHT: I'll inspect django/db/models/aggregates.py Count before I run anything.\n\n"
            "```mswea_bash_command\n"
            "grep Count /testbed/django/db/models/aggregates.py\n"
            "```"
        ),
        "extra": {
            "cost": 0.0,
            "actions": [{"command": "grep Count /testbed/django/db/models/aggregates.py"}],
        },
    }

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    message = agent.query()

    assert "I'll inspect django/db/models/aggregates.py Count before I run anything." in message["content"]
    assert "I'low_left" not in message["content"]
    assert "carry_out anything" not in message["content"]
    assert "```mswea_bash_command\ngrep Tally /testbed/working_repository/db/models/totals.py\n```" in message["content"]
    assert "working_repository" in message["extra"]["actions"][0]["command"]
    assert "totals.py" in message["extra"]["actions"][0]["command"]
    assert "Tally" in message["extra"]["actions"][0]["command"]


def test_execute_actions_uses_to_virtual_output_for_tool_results(mock_model, mock_env):
    """execute_actions() should sanitize runtime tool output via to_virtual_output()."""
    manager = MagicMock()
    manager.to_real.side_effect = lambda text: text
    manager.to_virtual.return_value = "WRONG"
    manager.to_virtual_output.return_value = "RIGHT"

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    mock_env.execute.return_value = {"output": "real output", "returncode": 0, "exception_info": ""}
    mock_model.format_observation_messages.side_effect = (
        lambda message, outputs, template_vars: [{"role": "tool", "content": outputs[0]["output"], "tool_call_id": "tc1"}]
    )

    message = {
        "content": "run command",
        "role": "assistant",
        "extra": {"actions": [{"command": "echo hi", "tool_call_id": "tc1"}]},
    }

    observations = agent.execute_actions(message)

    manager.to_virtual_output.assert_called_once_with("real output")
    manager.to_virtual.assert_not_called()
    assert observations[0]["content"] == "RIGHT"


def test_sanitize_tool_output_uses_error_output_sanitizer_for_tracebacks(mock_model, mock_env):
    """Runtime error output should not be over-virtualized."""
    manager = MagicMock()
    manager.to_virtual_error_output.return_value = "RIGHT_ERROR"
    manager.to_virtual_output.return_value = "WRONG"

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    result = agent._sanitize_tool_output(
        "Traceback (most recent call last):\nModuleNotFoundError: No module named 'django.utils'"
    )

    manager.to_virtual_error_output.assert_called_once()
    manager.to_virtual_output.assert_not_called()
    assert result == "RIGHT_ERROR"


def test_query_sanitizes_top_level_tool_calls(tmp_path, mock_model, mock_env):
    """query() should sanitize top-level tool_calls before assistant history is stored."""
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    mock_model.query.return_value = {
        "role": "assistant",
        "content": "Inspect django/db/models/aggregates.py Count",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "grep Count /testbed/django/db/models/aggregates.py"}',
                },
            }
        ],
        "extra": {
            "cost": 0.0,
            "actions": [{"command": "grep Count /testbed/django/db/models/aggregates.py", "tool_call_id": "call_1"}],
        },
    }

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    message = agent.query()

    tool_call_args = message["tool_calls"][0]["function"]["arguments"]
    assert "working_repository" in tool_call_args
    assert "totals.py" in tool_call_args
    assert "Tally" in tool_call_args
    assert "django" not in tool_call_args
    assert "aggregates.py" not in tool_call_args
    assert "Count" not in tool_call_args
    assert agent.messages[-1]["tool_calls"][0]["function"]["arguments"] == tool_call_args


def test_query_sanitizes_top_level_tool_calls_without_extra(tmp_path, mock_model, mock_env):
    """query() should sanitize top-level tool_calls even when extra is missing."""
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    mock_model.query.return_value = {
        "role": "assistant",
        "content": "Inspect django/db/models/aggregates.py Count",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "grep Count /testbed/django/db/models/aggregates.py"}',
                },
            }
        ],
    }

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    message = agent.query()

    tool_call_args = message["tool_calls"][0]["function"]["arguments"]
    assert "working_repository" in tool_call_args
    assert "totals.py" in tool_call_args
    assert "Tally" in tool_call_args
    assert agent.messages[-1]["tool_calls"][0]["function"]["arguments"] == tool_call_args


def test_query_does_not_add_tool_calls_key_when_absent(tmp_path, mock_model, mock_env):
    """query() should not materialize an empty tool_calls field for normal assistant turns."""
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    mock_model.query.return_value = {
        "role": "assistant",
        "content": "Inspect django/db/models/aggregates.py Count",
        "extra": {
            "cost": 0.0,
            "actions": [{"command": "grep Count /testbed/django/db/models/aggregates.py"}],
        },
    }

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    message = agent.query()

    assert "tool_calls" not in message
    assert "tool_calls" not in agent.messages[-1]


def test_query_preserves_response_style_text_but_sanitizes_function_calls(tmp_path, mock_model, mock_env):
    """query() should preserve response text while sanitizing function-call arguments/actions."""
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally", "run": "carry_out"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    from glasses.manager import SemanticMappingManager
    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    mock_model.query.return_value = {
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Inspect django/db/models/aggregates.py Count"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": '{"command": "grep Count /testbed/django/db/models/aggregates.py"}',
            },
            {
                "type": "reasoning",
                "text": "Reason about django/db/models/aggregates.py Count",
                "content": [
                    {"type": "output_text", "text": "Fallback django/db/models/aggregates.py Count"},
                ],
                "arguments": '{"command": "cat /testbed/django/db/models/aggregates.py | grep Count"}',
                "title": "Investigate django Count",
                "description": "Look in django/db/models/aggregates.py for Count",
                "query": "django aggregates.py Count",
            },
        ],
        "extra": {
            "cost": 0.0,
            "actions": [{"command": "grep Count /testbed/django/db/models/aggregates.py", "tool_call_id": "call_1"}],
        },
    }

    agent = SemanticMappingAgent(
        mock_model, mock_env,
        mapping_manager=manager,
        system_template="system",
        instance_template="{{task}}",
    )

    message = agent.query()

    assistant_text = message["output"][0]["content"][0]["text"]
    tool_call_args = message["output"][1]["arguments"]
    output_payload = json.dumps(message["output"])
    assert assistant_text == "Inspect django/db/models/aggregates.py Count"
    assert "working_repository" in tool_call_args
    assert "totals.py" in tool_call_args
    assert "Tally" in tool_call_args
    assert "Inspect django/db/models/aggregates.py Count" in output_payload
    assert "/testbed/working_repository/db/models/totals.py" in output_payload
    assert agent.messages[-1]["output"][0]["content"][0]["text"] == assistant_text
    assert agent.messages[-1]["output"][1]["arguments"] == tool_call_args
