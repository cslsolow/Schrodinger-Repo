"""Agent with real-time bidirectional semantic translation."""

import json
import logging
import re
from pathlib import Path

from minisweagent.agents.default import DefaultAgent

logger = logging.getLogger("mapped_agent")

_TEXT_COMMAND_BLOCK_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
_ERROR_OUTPUT_PATTERN = re.compile(
    r"(Traceback|ModuleNotFoundError|ImportError|No module named|<exception>|timed out after \d+ seconds)"
)


class SemanticMappingAgent(DefaultAgent):
    """Wraps DefaultAgent with real-time semantic mapping translation.

    When mapping_manager is provided:
    - run(): translates task description real -> virtual
    - execute_actions(): translates commands virtual -> real, outputs real -> virtual
    - serialize(): includes translator stats

    When mapping_manager is None, behaves identically to DefaultAgent.
    """

    def __init__(self, *args, mapping_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping_manager = mapping_manager
        self.raw_messages: list[dict] = []

    def _sanitize_assistant_message(self, message: dict) -> dict:
        if not self.mapping_manager:
            return message

        if message.get("object") == "response":
            return self._sanitize_response_message(message)

        if message.get("role") != "assistant":
            return message

        sanitized = dict(message)
        if "content" in sanitized:
            sanitized["content"] = self._sanitize_assistant_content(sanitized["content"])

        if "tool_calls" in sanitized:
            sanitized["tool_calls"] = [self._sanitize_tool_call(tool_call) for tool_call in sanitized.get("tool_calls", [])]

        extra = sanitized.get("extra")
        if isinstance(extra, dict):
            sanitized_extra = dict(extra)
            sanitized_extra["actions"] = self._sanitize_actions(sanitized_extra.get("actions", []))
            sanitized["extra"] = sanitized_extra
        return sanitized

    def _sanitize_assistant_content(self, content):
        if not isinstance(content, str):
            return content

        # Split on command blocks (capturing group keeps the command text in parts)
        parts = _TEXT_COMMAND_BLOCK_RE.split(content)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                result.append(part)
            else:
                virtual_command = self.mapping_manager.to_virtual_command(part)
                result.append(f"```mswea_bash_command\n{virtual_command}\n```")
        return "".join(result)

    def _sanitize_response_message(self, message: dict) -> dict:
        sanitized = dict(message)
        sanitized["output"] = [self._sanitize_response_output_item(item) for item in sanitized.get("output", [])]

        extra = sanitized.get("extra")
        if isinstance(extra, dict):
            sanitized_extra = dict(extra)
            sanitized_extra["actions"] = self._sanitize_actions(sanitized_extra.get("actions", []))
            sanitized["extra"] = sanitized_extra
        return sanitized

    def _sanitize_response_output_item(self, item: dict) -> dict:
        item_type = item.get("type")
        if item_type == "function_call":
            return self._sanitize_response_function_call(item)
        if item_type != "message":
            return self._sanitize_response_fallback(item)

        sanitized = dict(item)
        sanitized["content"] = [self._sanitize_response_content_item(content) for content in item.get("content", [])]
        return sanitized

    def _sanitize_response_fallback(self, value, field: str | None = None):
        if isinstance(value, dict):
            return {key: self._sanitize_response_fallback(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_response_fallback(item, field) for item in value]
        if isinstance(value, str):
            if field == "arguments":
                return self._sanitize_arguments_string(value)
            return value
        return value

    def _sanitize_response_content_item(self, item: dict) -> dict:
        return item

    def _sanitize_response_function_call(self, tool_call: dict) -> dict:
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, str):
            return tool_call

        return {
            **tool_call,
            "arguments": self._sanitize_arguments_string(arguments),
        }

    def _sanitize_arguments_string(self, arguments: str) -> str:
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return self.mapping_manager.to_virtual_command(arguments)
        if isinstance(parsed_arguments, dict) and isinstance(parsed_arguments.get("command"), str):
            parsed_arguments["command"] = self.mapping_manager.to_virtual_command(parsed_arguments["command"])
            return json.dumps(parsed_arguments)
        return self.mapping_manager.to_virtual_command(arguments)

    def _sanitize_actions(self, actions: list[dict]) -> list[dict]:
        return [
            {
                **action,
                "command": self.mapping_manager.to_virtual_command(action.get("command", "")),
            }
            for action in actions
        ]

    def _sanitize_tool_call(self, tool_call: dict) -> dict:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return tool_call

        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return tool_call

        return {
            **tool_call,
            "function": {
                **function,
                "arguments": self._sanitize_arguments_string(arguments),
            },
        }

    def run(self, task: str = "", **kwargs) -> dict:
        self.raw_messages = []
        if self.mapping_manager:
            task = self.mapping_manager.to_virtual_text(task)
        return super().run(task, **kwargs)

    def query(self) -> dict:
        message = super().query()
        if not self.mapping_manager:
            return message

        self.raw_messages.append({"type": "query_raw", "message": message})
        sanitized = self._sanitize_assistant_message(message)
        self.messages[-1] = sanitized
        return sanitized

    def execute_actions(self, message: dict) -> list[dict]:
        if not self.mapping_manager:
            return super().execute_actions(message)

        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        raw_action_records = []

        for action in actions:
            virtual_command = action.get("command", "")
            # Translate command: virtual -> real
            real_command = self.mapping_manager.to_real_command(virtual_command)
            real_action = {**action, "command": real_command}

            logger.debug(f"Translated command: {virtual_command!r} -> {real_command!r}")

            # Execute in real space
            output = self.env.execute(real_action)
            real_output = output.get("output")

            raw_action_records.append({
                "virtual_command": virtual_command,
                "real_command": real_command,
                "real_output": real_output,
            })

            # Translate output: real -> virtual
            if "output" in output:
                output["output"] = self._sanitize_tool_output(output["output"])

            outputs.append(output)

        if raw_action_records:
            self.raw_messages.append({"type": "execute_raw", "actions": raw_action_records})

        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    def _sanitize_tool_output(self, output):
        if not isinstance(output, str):
            return output
        if _ERROR_OUTPUT_PATTERN.search(output):
            return self.mapping_manager.to_virtual_error_output(output)
        return self.mapping_manager.to_virtual_output(output)

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        if self.mapping_manager:
            data["info"]["translator_stats"] = self.mapping_manager.get_translator_stats()
            data["raw_messages"] = self.raw_messages
        return data
