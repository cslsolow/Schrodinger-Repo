import ast
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import litellm

from glasses.level4_semantic_rewrite import is_protected_path
from minisweagent.models import get_model

MODE = "llm_function_body_rewrite"


@dataclass(frozen=True)
class RewriteTarget:
    path: Path
    function_name: str
    start_line: int
    end_line: int


def _node_end_lineno(node: ast.AST) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is not None:
        return end_lineno
    end_lineno = getattr(node, "lineno", 0)
    for child in ast.iter_child_nodes(node):
        end_lineno = max(end_lineno, _node_end_lineno(child))
    return end_lineno


def _smallest_enclosing_function(source: str, line_no: int):
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_lineno = _node_end_lineno(node)
            if node.lineno <= line_no <= end_lineno:
                matches.append(node)
    if not matches:
        return None
    return min(matches, key=lambda node: (_node_end_lineno(node) - node.lineno, node.lineno))


def _function_nodes(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _top_level_function_nodes(source: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _resolve_patch_path(raw_path: str, repo_root: Path):
    path = Path(raw_path)
    if path.is_absolute():
        return None
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return resolved


def _extract_patch_files(patch_text: str):
    files = []
    current_path = None
    current_hunks = None
    current_lines = None
    current_line_no = None
    for line in patch_text.splitlines():
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if match:
            old_path, new_path = match.groups()
            if current_hunks is not None:
                if current_lines is not None:
                    current_hunks.append(current_lines)
                files.append((current_path, current_hunks))
            current_path = Path(old_path if new_path == "/dev/null" else new_path)
            current_hunks = []
            current_lines = None
            current_line_no = None
            continue
        if current_hunks is None:
            continue
        if line.startswith("@@ "):
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            line_no = int(match.group(1)) if match else None
            if current_lines is not None:
                current_hunks.append(current_lines)
            current_lines = []
            current_line_no = line_no
            continue
        if current_lines is None or current_line_no is None:
            continue
        prefix = line[:1]
        if prefix == "-":
            current_lines.append(current_line_no)
        elif prefix == "+":
            current_lines.append(current_line_no)
            current_line_no += 1
        elif prefix == " ":
            current_lines.append(current_line_no)
            current_line_no += 1
    if current_hunks is not None:
        if current_lines is not None:
            current_hunks.append(current_lines)
        files.append((current_path, current_hunks))
    return files


def extract_rewrite_targets_from_golden_patch(patch_text: str, repo_root: Path):
    targets = []
    seen = set()
    for path, hunks in _extract_patch_files(patch_text):
        if is_protected_path(str(path)):
            continue
        source_path = _resolve_patch_path(str(path), repo_root)
        if source_path is None or not source_path.exists():
            continue
        source = source_path.read_text()
        for hunk in hunks:
            matches = []
            seen_in_hunk = set()
            for line_no in hunk:
                if line_no is None:
                    continue
                node = _smallest_enclosing_function(source, line_no)
                if node is not None:
                    key = (node.name, node.lineno, _node_end_lineno(node))
                    if key in seen_in_hunk:
                        continue
                    seen_in_hunk.add(key)
                    matches.append(node)
            if not matches:
                continue
            for node in sorted(matches, key=lambda item: (item.lineno, _node_end_lineno(item), item.name)):
                target = RewriteTarget(path=path, function_name=node.name, start_line=node.lineno, end_line=_node_end_lineno(node))
                key = (target.path, target.function_name, target.start_line, target.end_line)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(target)

    if not targets:
        return [], "no_rewriteable_function_body"
    return sorted(targets, key=lambda item: (item.path.as_posix(), item.start_line, item.function_name)), None


def _resolve_test_module_path(repo_root: Path, test_id: str):
    parts = [part for part in (test_id or "").split(".") if part]
    if len(parts) < 2:
        return None, None
    for module_end in range(len(parts) - 1, 0, -1):
        module_parts = parts[:module_end]
        qualname_parts = parts[module_end:]
        rel_module = Path(*module_parts)
        candidates = [rel_module.with_suffix(".py"), rel_module / "__init__.py"]
        for candidate in candidates:
            source_path = repo_root / candidate
            if source_path.exists():
                normalized = list(qualname_parts)
                normalized[-1] = normalized[-1].split("[", 1)[0]
                return candidate, normalized
    return None, None


def _find_qualname_node(source: str, qualname_parts: list[str]):
    if not qualname_parts:
        return None
    tree = ast.parse(source)
    body = tree.body
    node = None
    for index, name in enumerate(qualname_parts):
        if index == len(qualname_parts) - 1:
            allowed = (ast.FunctionDef, ast.AsyncFunctionDef)
        else:
            allowed = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        node = next((item for item in body if isinstance(item, allowed) and item.name == name), None)
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node


def _normalized_source_segment(source: str, node: ast.AST) -> str | None:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        lines = source.splitlines(keepends=True)
        start = getattr(node, "lineno", 1) - 1
        end = _node_end_lineno(node)
        segment = "".join(lines[start:end])
    lines = segment.splitlines(keepends=True)
    indent = " " * getattr(node, "col_offset", 0)
    if indent:
        normalized_lines = []
        for line in lines:
            if line.startswith(indent):
                normalized_lines.append(line[len(indent) :])
            else:
                normalized_lines.append(line)
        normalized = "".join(normalized_lines)
    else:
        normalized = segment
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def extract_pass_to_pass_test_sources(repo_root: Path, test_ids: list[str]) -> list[dict]:
    snippets = []
    seen = set()
    for test_id in test_ids or []:
        rel_path, qualname_parts = _resolve_test_module_path(repo_root, test_id)
        if rel_path is None or not qualname_parts:
            continue
        key = (rel_path.as_posix(), tuple(qualname_parts))
        if key in seen:
            continue
        seen.add(key)
        source = (repo_root / rel_path).read_text()
        node = _find_qualname_node(source, qualname_parts)
        if node is None:
            continue
        snippets.append(
            {
                "test_id": test_id,
                "path": rel_path.as_posix(),
                "source": _normalized_source_segment(source, node),
            }
        )
    return snippets


def _selected_function_node(source: str, target):
    if target is not None:
        target_name, start_line = target[0], target[1]
        for node in _function_nodes(source):
            if node.name == target_name and node.lineno == start_line:
                return node
        return None
    nodes = list(_top_level_function_nodes(source))
    if len(nodes) == 1:
        return nodes[0]
    return None


def _dump_optional_ast(node):
    if node is None:
        return None
    return ast.dump(node, include_attributes=False)


def _selected_exact_function_node(source: str, target):
    if target is None:
        return None
    target_name, start_line = target[0], target[1]
    for node in _function_nodes(source):
        if node.name == target_name and node.lineno == start_line:
            return node
    return None


def _function_span_start_lineno(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    decorator_lines = [getattr(decorator, "lineno", getattr(node, "lineno", 0)) for decorator in decorators]
    if decorator_lines:
        return min(decorator_lines)
    return getattr(node, "lineno", 0)


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _indent_replacement_source(source: str, target_indent: str) -> str:
    trailing_newline = source.endswith("\n")
    lines = textwrap.dedent(source).splitlines()
    indented = [target_indent + line if line.strip() else line for line in lines]
    return "\n".join(indented) + ("\n" if trailing_newline else "")


def _strip_code_fence(content: str) -> str:
    text = (content or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1]
    return text.strip()


def build_level4b_prompt(
    *,
    path: str,
    function_name: str,
    original_function_source: str,
    enclosing_context: str,
    failure_feedback: str | None = None,
) -> str:
    prompt = (
        "Rewrite the target function so the result is a substantially different implementation "
        "while preserving behavior, inputs, outputs, and public API.\n\n"
        f"Target path: {path}\n"
        f"Target function: {function_name}\n\n"
        "Rules:\n"
        "- do not change the function name\n"
        "- do not change arguments\n"
        "- do not change decorators\n"
        "- do not change the return annotation\n"
        "- do not change imports\n"
        "- do not modify code outside this function\n"
        "- do not make a trivial rewrite\n\n"
        "Return only JSON with these keys:\n"
        "- rewritten_function_source\n"
        "- rewrite_summary\n"
        "- behavioral_invariants\n\n"
        "Original function:\n"
        f"```python\n{original_function_source}```\n\n"
        "Enclosing context:\n"
        f"```python\n{enclosing_context}```"
    )
    if failure_feedback:
        prompt += f"\n\nPrevious attempt failed. Avoid repeating these issues:\n{failure_feedback}"
    return prompt


def parse_level4b_response(content: str) -> dict:
    try:
        data = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid level4b response: malformed JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("Invalid level4b response: expected a JSON object")

    required_keys = ("rewritten_function_source", "rewrite_summary", "behavioral_invariants")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"Invalid level4b response: missing keys: {', '.join(missing)}")

    rewritten_function_source = data["rewritten_function_source"]
    rewrite_summary = data["rewrite_summary"]
    behavioral_invariants = data["behavioral_invariants"]

    if not isinstance(rewritten_function_source, str):
        raise ValueError("Invalid level4b response: rewritten_function_source must be a string")
    if not isinstance(rewrite_summary, str):
        raise ValueError("Invalid level4b response: rewrite_summary must be a string")
    if not isinstance(behavioral_invariants, list) or not all(isinstance(item, str) for item in behavioral_invariants):
        raise ValueError("Invalid level4b response: behavioral_invariants must be a list of strings")

    return {
        "rewritten_function_source": rewritten_function_source,
        "rewrite_summary": rewrite_summary,
        "behavioral_invariants": behavioral_invariants,
    }


def generate_level4b_rewrite(
    *,
    model_name: str,
    model_config: dict,
    path: str,
    function_name: str,
    original_function_source: str,
    enclosing_context: str,
    failure_feedback: str | None = None,
) -> dict:
    model = get_model(model_name, model_config)
    model_kwargs = dict(getattr(model.config, "model_kwargs", {}))
    model_kwargs.pop("parallel_tool_calls", None)
    prompt = build_level4b_prompt(
        path=path,
        function_name=function_name,
        original_function_source=original_function_source,
        enclosing_context=enclosing_context,
        failure_feedback=failure_feedback,
    )
    raw = litellm.completion(
        model=model.config.model_name,
        messages=[{"role": "user", "content": prompt}],
        **model_kwargs,
    )
    return parse_level4b_response(raw.choices[0].message.content or "")


def target_group_item_key(item: dict) -> str:
    return f"{item['path']}:{item['start_line']}:{item['function_name']}"


def serialize_target_group(targets: list[RewriteTarget]) -> list[dict]:
    return sorted(
        [
            {
                "path": target.path.as_posix(),
                "function_name": target.function_name,
                "start_line": target.start_line,
                "end_line": target.end_line,
            }
            for target in targets
        ],
        key=target_group_item_key,
    )


def build_level4b_runtime_record(
    *,
    seed: int,
    runtime_seed: int,
    variant_name: str,
    rewritten_files: list[str],
    target_group: list[dict],
    target_group_signature: str,
    rewrite_summaries: dict[str, str],
    behavioral_invariants: dict[str, list[str]],
    rewrite_strength: float,
    status: str = "applied",
) -> dict:
    return {
        "mode": MODE,
        "status": status,
        "seed": seed,
        "runtime_seed": runtime_seed,
        "variant_name": variant_name,
        "rewritten_files": rewritten_files,
        "target_group": target_group,
        "target_group_size": len(target_group),
        "target_group_signature": target_group_signature,
        "rewrite_summaries": rewrite_summaries,
        "behavioral_invariants": behavioral_invariants,
        "rewrite_strength": rewrite_strength,
    }


def replace_function_in_source(source: str, function_name: str, rewritten_function_source: str, target=None) -> str:
    if target is not None:
        node = _selected_exact_function_node(source, target)
        if node is None or node.name != function_name:
            raise ValueError("target function not found for exact target tuple")
    else:
        node = _selected_function_node(source, target)
        if node is None or node.name != function_name:
            nodes = [item for item in _function_nodes(source) if item.name == function_name]
            if not nodes:
                return source
            node = min(nodes, key=lambda item: (item.lineno, _node_end_lineno(item)))

    lines = source.splitlines(keepends=True)
    start = _function_span_start_lineno(node) - 1
    end = _node_end_lineno(node)
    replacement = _indent_replacement_source(rewritten_function_source, _line_indent(lines[start]))
    if replacement and not replacement.endswith("\n"):
        replacement += "\n"
    lines[start:end] = replacement.splitlines(keepends=True)
    return "".join(lines)


def validate_rewritten_function_body(original_source: str, rewritten_source: str, target=None):
    try:
        original = _selected_function_node(original_source, target)
        rewritten = _selected_function_node(rewritten_source, target)
    except SyntaxError:
        return False, "syntax_error"

    if original is None or rewritten is None:
        return False, "syntax_error"

    if type(original) is not type(rewritten):
        return False, "function_type_changed"

    if original.name != rewritten.name:
        return False, "function_name_changed"
    if ast.dump(original.args, include_attributes=False) != ast.dump(rewritten.args, include_attributes=False):
        return False, "function_signature_changed"
    if [ast.dump(node, include_attributes=False) for node in original.decorator_list] != [
        ast.dump(node, include_attributes=False) for node in rewritten.decorator_list
    ]:
        return False, "decorators_changed"
    if _dump_optional_ast(original.returns) != _dump_optional_ast(rewritten.returns):
        return False, "return_annotation_changed"

    try:
        ast.parse(rewritten_source)
    except SyntaxError:
        return False, "syntax_error"
    return True, None


def rewrite_strength_gate(original_source: str, rewritten_source: str, target=None):
    try:
        original = _selected_function_node(original_source, target)
        rewritten = _selected_function_node(rewritten_source, target)
    except SyntaxError:
        return False, "syntax_error"

    if original is None or rewritten is None:
        return False, "syntax_error"

    if len(original.body) == 1 and len(rewritten.body) == 2:
        first, second = rewritten.body
        if (
            isinstance(first, ast.Assign)
            and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and isinstance(second, ast.Return)
            and isinstance(second.value, ast.Name)
            and second.value.id == first.targets[0].id
        ):
            return False, "rewrite_too_trivial"

    return True, None
