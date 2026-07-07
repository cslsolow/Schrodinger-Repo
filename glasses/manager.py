import json
import re
import time
from pathlib import Path

from glasses.mapping_bundle import compose_forward_map, load_mapping_bundle


_COMMAND_SPLIT_PATTERN = re.compile(r"(\|\||&&|;)")
_ENV_ASSIGN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^\s]+$")
_COMMAND_PART_PATTERN = re.compile(r"^(\s*(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*)((?:\S+)?)(.*)$")
_PYTHON_C_PATTERN = re.compile(r"(?P<prefix>\s+-c\s+)(?P<quote>['\"])(?P<code>.*?)(?P=quote)", re.DOTALL)
_SLASH_PATH_PATTERN = re.compile(r"/?[\w.-]+(?:/[\w.-]+)+")
_DOTTED_PATH_PATTERN = re.compile(
    r"(?<![/\w-])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?![/\w-])"
)


class SemanticMappingManager:
    def __init__(
        self,
        instance_dir: Path,
        seed: int,
        debug_log_path: Path | None = None,
        enabled_layers: list[str] | None = None,
        project_name: str | None = None,
    ):
        self.instance_dir = instance_dir
        self.translate_count = 0
        self.bundle = None
        self.enabled_layers = enabled_layers or []
        self.project_name = project_name
        self.strict_session_reverse = False

        if self._has_mapping_bundle(instance_dir):
            self.bundle = load_mapping_bundle(instance_dir)
            self.strict_session_reverse = True
            bundle_layers = self.bundle["mapping_meta"].get("enabled_layers", [])
            self.enabled_layers = bundle_layers if enabled_layers is None else enabled_layers
            self.forward_map = self._build_runtime_forward_map()
            self.virtual_project_name = self.bundle["identity_map"].get(self.project_name) if self.project_name else None
        else:
            mapping_path = instance_dir / f"mapping_seed_{seed}.json"
            self.forward_map = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
            self.virtual_project_name = None

        self.backward_map = {v: k for k, v in self.forward_map.items()}

        # Load token-level mapping (for generation phase only, not used at runtime)
        token_path = instance_dir / "token_mapping.json"
        self.token_forward = json.loads(token_path.read_text()) if token_path.exists() else {}
        self.token_backward = {v: k for k, v in self.token_forward.items()}

        # Session notebook: tracks what to_virtual() actually replaced
        # Key: virtual_name, Value: real_name
        # Accumulates across the entire instance lifetime
        self.session_notebook: dict[str, str] = {}
        self.session_path_notebook: dict[str, str] = {}

        # Debug log path
        self.debug_log_path = debug_log_path or (instance_dir / "translator_debug.jsonl")

        self.sacrosanct_words = {
            "class", "def", "if", "else", "elif", "try", "except", "finally",
            "yield", "return", "import", "from", "as", "for", "in", "while",
            "break", "continue", "raise", "is", "not", "and", "or",
            "none", "true", "false", "async", "await", "pass", "global",
            "nonlocal", "assert", "lambda", "del", "super", "isinstance",
            "type", "str", "int", "float", "list", "dict", "set", "tuple",
            "bool", "object", "open", "len", "print", "diff", "index",
            "---", "+++", "@@", "git", "a", "b",
            "__init__", "__str__", "__repr__", "__dict__", "__class__",
            "__module__", "__name__", "__file__",
            # Common English words that appear as repo method/var names but must not pollute prose
            "to", "do", "of", "by", "on", "at", "up", "no", "so", "be",
            "go", "me", "my", "he", "we", "it", "ok", "an", "if", "id",
            "run", "execute", "process", "lift", "get", "set", "add", "put",
            "use", "new", "old", "out", "off", "all", "any", "may", "can",
            # Extended: common English words found in namespace_symbol_map
            "bar", "base", "body", "box", "case", "cast", "cell", "char",
            "city", "copy", "core", "date", "each", "edge", "end", "find",
            "flag", "form", "func", "hook", "init", "item", "join", "kind",
            "last", "left", "link", "load", "make", "meta", "move", "node",
            "only", "pack", "page", "pick", "pipe", "plot", "post", "push",
            "read", "real", "rule", "save", "scan", "send", "show", "sign",
            "size", "skip", "sort", "stop", "swap", "sync", "tail", "test",
            "tree", "trim", "unit", "view", "walk", "wrap",
        }

        # Code / command translation must preserve exact case.
        self._fwd_id_pattern, self._fwd_id_lookup = self._compile_exact_pattern(self.forward_map)
        self._bwd_id_pattern, self._bwd_id_lookup = self._compile_exact_pattern(self.backward_map)
        # Text sanitization may still use case-folded replacements when unambiguous.
        self._fwd_folded_pattern, self._fwd_folded_lookup = self._compile_folded_pattern(self.forward_map)
        self._namespace_path_rules = self._build_namespace_path_rules()
        self._reverse_namespace_path_rules = {v: k for k, v in self._namespace_path_rules.items()}
        self._namespace_path_rules_folded = {k.lower(): v for k, v in self._namespace_path_rules.items()}
        self._reverse_namespace_path_rules_folded = {k.lower(): v for k, v in self._reverse_namespace_path_rules.items()}
        self._register_project_identity_variants()

    def _has_mapping_bundle(self, instance_dir: Path):
        bundle_files = (
            "identity_map.json",
            "namespace_symbol_map.json",
            "namespace_path_map.json",
            "mapping_meta.json",
        )
        return all((instance_dir / name).exists() for name in bundle_files)

    def _build_runtime_forward_map(self):
        bundle = dict(self.bundle)
        bundle["namespace_path_map"] = {}
        return compose_forward_map(bundle, self.enabled_layers)

    def _build_namespace_path_rules(self):
        rules = {}
        if self.bundle and "namespace_l2" in self.enabled_layers:
            rules.update(self.bundle["namespace_path_map"])
        if self.project_name and self.virtual_project_name and "namespace_l2" in self.enabled_layers:
            rules.setdefault(self.project_name, self.virtual_project_name)
        return rules

    def _register_project_identity_variants(self):
        self._output_sanitize_rules = []
        self._text_identity_sanitize_rules = []
        self._error_identity_sanitize_rules = []
        if not self.project_name or not self.virtual_project_name:
            return
        if "identity_l1" not in self.enabled_layers:
            return
        bare_repo = re.compile(rf"(?<![\w-]){re.escape(self.project_name)}(?![\w-])", re.IGNORECASE)
        self._output_sanitize_rules.append((bare_repo, self.virtual_project_name))
        self._error_identity_sanitize_rules.extend(
            [
                (
                    re.compile(rf"(?<=/testbed/){re.escape(self.project_name)}(?=/)", re.IGNORECASE),
                    self.virtual_project_name,
                ),
                (
                    re.compile(rf"(?<=\./){re.escape(self.project_name)}(?=/)", re.IGNORECASE),
                    self.virtual_project_name,
                ),
            ]
        )
        self._text_identity_sanitize_rules.extend(
            [
                (bare_repo, self.virtual_project_name),
                (
                    re.compile(rf"(?<![\w-]){re.escape(self.project_name)}(?=_)", re.IGNORECASE),
                    self.virtual_project_name,
                ),
                (
                    re.compile(rf"(?<![\w-]){re.escape(self.project_name)}(?=[0-9-])", re.IGNORECASE),
                    self.virtual_project_name,
                ),
                (
                    re.compile(rf"(?<=\.){re.escape(self.project_name)}(?:project)?(?=\.)", re.IGNORECASE),
                    self.virtual_project_name,
                ),
            ]
        )

    def _filter_mapping(self, mapping: dict[str, str]) -> dict[str, str]:
        return {
            k: v for k, v in mapping.items()
            if len(k) >= 2 and k.lower() not in self.sacrosanct_words and k != v
        }

    def _compile_exact_pattern(self, mapping: dict[str, str]):
        filtered = self._filter_mapping(mapping)
        if not filtered:
            return None, {}
        sorted_keys = sorted(filtered.keys(), key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b")
        return pattern, filtered

    def _compile_folded_pattern(self, mapping: dict[str, str]):
        """Build a case-insensitive pattern only for unambiguous lowercase keys."""
        filtered = self._filter_mapping(mapping)
        if not filtered:
            return None, {}
        folded: dict[str, str] = {}
        ambiguous: set[str] = set()
        for key, value in filtered.items():
            lowered = key.lower()
            if lowered in folded and folded[lowered] != value:
                ambiguous.add(lowered)
                continue
            folded[lowered] = value
        for lowered in ambiguous:
            folded.pop(lowered, None)
        if not folded:
            return None, {}
        sorted_keys = sorted(folded.keys(), key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b", re.IGNORECASE)
        return pattern, folded

    def _compile_pattern(self, mapping: dict[str, str]):
        """Backward-compatible alias for existing tests that expect folded behavior in text."""
        filtered = {k: v for k, v in mapping.items()
                    if len(k) >= 2 and k.lower() not in self.sacrosanct_words and k != v}
        if not filtered:
            return None, {}
        sorted_keys = sorted(filtered.keys(), key=len, reverse=True)
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b', re.IGNORECASE)
        return pattern, {k.lower(): v for k, v in filtered.items()}

    def _replace_identifiers_exact(self, text: str, pattern, lookup, *, record_notebook: bool) -> str:
        if not isinstance(text, str) or not text or not pattern or not lookup:
            return text

        def _replace_and_record(match):
            real_name = match.group()
            virtual_name = lookup.get(real_name)
            if virtual_name:
                if record_notebook:
                    self.session_notebook[virtual_name] = real_name
                return virtual_name
            return real_name

        return pattern.sub(_replace_and_record, text)

    def _replace_identifiers_folded(self, text: str, pattern, lookup, *, record_notebook: bool) -> str:
        if not isinstance(text, str) or not text or not pattern or not lookup:
            return text

        def _replace_and_record(match):
            real_name = match.group()
            virtual_name = lookup.get(real_name.lower())
            if virtual_name:
                if record_notebook:
                    self.session_notebook[virtual_name] = real_name
                return virtual_name
            return real_name

        return pattern.sub(_replace_and_record, text)

    def to_virtual(self, text: str) -> str:
        """Translate real -> virtual. Records all actual replacements in session_notebook."""
        if not isinstance(text, str) or not text:
            return text

        original_text = text
        text = self._replace_identifiers_exact(
            text,
            self._fwd_id_pattern,
            self._fwd_id_lookup,
            record_notebook=True,
        )
        self.translate_count += 1
        self._log_translation(original_text, text, to_real=False)
        return text

    def to_virtual_output(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return self._to_virtual_core(text)

    def to_virtual_error_output(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        text = self._rewrite_repo_owned_error_tokens(text)
        for pattern, repl in self._error_identity_sanitize_rules:
            text = pattern.sub(repl, text)
        return text

    def to_virtual_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        text = self._to_virtual_core(text)
        for pattern, repl in self._text_identity_sanitize_rules:
            text = pattern.sub(repl, text)
        return text

    def to_virtual_command(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return self._rewrite_command_preserving_head(text, self._to_virtual_command_core)

    def _to_virtual_core(self, text: str) -> str:
        text = self._rewrite_paths_and_symbols(
            text,
            path_rules=self._namespace_path_rules,
            path_rules_folded=self._namespace_path_rules_folded,
            symbol_pattern=self._fwd_folded_pattern,
            symbol_lookup=self._fwd_folded_lookup,
            record_notebook=True,
            folded_symbols=True,
        )
        for pattern, repl in self._output_sanitize_rules:
            text = pattern.sub(repl, text)
        return text

    def _to_virtual_command_core(self, text: str) -> str:
        return self._rewrite_paths_and_symbols(
            text,
            path_rules=self._namespace_path_rules,
            path_rules_folded=self._namespace_path_rules_folded,
            symbol_pattern=self._fwd_id_pattern,
            symbol_lookup=self._fwd_id_lookup,
            record_notebook=True,
            folded_symbols=False,
        )

    def _lookup_path_rule(
        self,
        segment: str,
        rules_exact: dict[str, str],
        rules_folded: dict[str, str],
    ) -> tuple[str, str | None]:
        replacement = rules_exact.get(segment)
        if replacement is not None:
            return replacement, segment

        lowered = segment.lower()
        replacement = rules_folded.get(lowered)
        if replacement is None:
            return segment, None

        canonical = next((key for key in rules_exact if key.lower() == lowered), segment)
        return replacement, canonical

    def _record_path_notebook(self, virtual_segment: str, real_segment: str | None) -> None:
        if real_segment is not None and virtual_segment != real_segment:
            self.session_path_notebook[virtual_segment] = real_segment

    def _replace_path_segments(
        self,
        token: str,
        rules_exact: dict[str, str],
        rules_folded: dict[str, str],
        *,
        record_notebook: bool,
    ) -> str:
        leading_slash = token.startswith("/")
        segments = token.split("/")
        rewritten = []
        for segment in segments:
            replacement, real_segment = self._lookup_path_rule(segment, rules_exact, rules_folded)
            if record_notebook:
                self._record_path_notebook(replacement, real_segment)
            rewritten.append(replacement)
        result = "/".join(rewritten)
        if leading_slash and not result.startswith("/"):
            return "/" + result
        return result

    def _replace_module_segments(
        self,
        token: str,
        rules_exact: dict[str, str],
        rules_folded: dict[str, str],
        *,
        record_notebook: bool,
    ) -> str:
        rewritten = []
        for segment in token.split("."):
            replacement, real_segment = self._lookup_path_rule(segment, rules_exact, rules_folded)
            if record_notebook:
                self._record_path_notebook(replacement, real_segment)
            rewritten.append(replacement)
        return ".".join(rewritten)

    def _rewrite_paths(
        self,
        text: str,
        rules_exact: dict[str, str],
        rules_folded: dict[str, str],
        *,
        record_notebook: bool,
    ) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}

        def _stash(value: str) -> str:
            key = f"__MSWEA_PATH_{len(placeholders)}__"
            placeholders[key] = value
            return key

        text = _SLASH_PATH_PATTERN.sub(
            lambda m: _stash(
                self._replace_path_segments(
                    m.group(),
                    rules_exact,
                    rules_folded,
                    record_notebook=record_notebook,
                )
            ),
            text,
        )

        bare_file_rules = {k: v for k, v in rules_exact.items() if "." in k and "/" not in k}
        if bare_file_rules:
            bare_file_names = sorted(bare_file_rules, key=len, reverse=True)
            bare_file_pattern = re.compile(
                r"(?<![/\w-])(" + "|".join(re.escape(name) for name in bare_file_names) + r")(?![/\w-])"
            )
            def _replace_bare_file(match):
                real_name = match.group()
                virtual_name = bare_file_rules.get(real_name, real_name)
                if record_notebook:
                    self._record_path_notebook(virtual_name, real_name)
                return _stash(virtual_name)

            text = bare_file_pattern.sub(_replace_bare_file, text)
        return text, placeholders

    def _restore_placeholders(self, text: str, placeholders: dict[str, str]) -> str:
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    def _rewrite_dotted_paths(
        self,
        text: str,
        rules_exact: dict[str, str],
        rules_folded: dict[str, str],
        *,
        record_notebook: bool,
    ) -> str:
        dotted_rules = {k: v for k, v in rules_exact.items() if "." not in k}
        dotted_rules_folded = {k.lower(): v for k, v in dotted_rules.items()}
        return _DOTTED_PATH_PATTERN.sub(
            lambda m: self._replace_module_segments(
                m.group(),
                dotted_rules,
                dotted_rules_folded,
                record_notebook=record_notebook,
            ),
            text,
        )

    def _rewrite_paths_and_symbols(
        self,
        text: str,
        *,
        path_rules: dict[str, str],
        path_rules_folded: dict[str, str],
        symbol_pattern,
        symbol_lookup,
        record_notebook: bool,
        folded_symbols: bool,
    ) -> str:
        text, placeholders = self._rewrite_paths(
            text,
            path_rules,
            path_rules_folded,
            record_notebook=record_notebook,
        )
        if folded_symbols:
            text = self._replace_identifiers_folded(
                text,
                symbol_pattern,
                symbol_lookup,
                record_notebook=record_notebook,
            )
        else:
            text = self._replace_identifiers_exact(
                text,
                symbol_pattern,
                symbol_lookup,
                record_notebook=record_notebook,
            )
        text = self._restore_placeholders(text, placeholders)
        return self._rewrite_dotted_paths(
            text,
            path_rules,
            path_rules_folded,
            record_notebook=record_notebook,
        )

    def _rewrite_path_like_tokens(self, text: str, rules: dict[str, str]) -> str:
        if not rules:
            return text
        lowered_rules = {k.lower(): v for k, v in rules.items()}

        def _replace_path(match):
            token = match.group()
            leading_slash = token.startswith("/")
            segments = token.split("/")
            rewritten = [lowered_rules.get(segment.lower(), segment) for segment in segments]
            result = "/".join(rewritten)
            if leading_slash and not result.startswith("/"):
                return "/" + result
            return result

        text = re.sub(r"/?[\w.-]+(?:/[\w.-]+)+", _replace_path, text)

        bare_file_rules = {k: v for k, v in rules.items() if "." in k and "/" not in k}
        if bare_file_rules:
            bare_file_names = sorted(bare_file_rules, key=len, reverse=True)
            bare_file_pattern = re.compile(
                r"(?<![/\w-])(" + "|".join(re.escape(name) for name in bare_file_names) + r")(?![/\w-])",
                re.IGNORECASE,
            )
            text = bare_file_pattern.sub(lambda m: lowered_rules.get(m.group().lower(), m.group()), text)

        # Rewrite dotted module paths such as django.db.models -> working_repository.storage_engine.object_models.
        # This closes the gap between slash-path translation and Python import/module references.
        def _replace_module_path(match):
            token = match.group()
            segments = token.split(".")
            rewritten = [lowered_rules.get(segment.lower(), segment) for segment in segments]
            return ".".join(rewritten)

        return re.sub(
            r"(?<![/\w-])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?![/\w-])",
            _replace_module_path,
            text,
        )

    def _rewrite_repo_owned_error_tokens(self, text: str) -> str:
        if not self.project_name or not self.virtual_project_name:
            return text

        def _rewrite_segment(segment: str) -> str:
            replacement, real_segment = self._lookup_path_rule(
                segment,
                self._namespace_path_rules,
                self._namespace_path_rules_folded,
            )
            self._record_path_notebook(replacement, real_segment)
            return replacement

        def _rewrite_segments(tail: str, sep: str) -> str:
            if not tail:
                return tail
            return sep.join(_rewrite_segment(segment) for segment in tail.split(sep))

        def _replace_rooted_path(match):
            prefix = match.group("prefix")
            tail = match.group("tail")
            self._record_path_notebook(self.virtual_project_name, self.project_name)
            rewritten_tail = _rewrite_segments(tail, "/")
            return f"{prefix}{self.virtual_project_name}/{rewritten_tail}"

        text = re.sub(
            rf"(?P<prefix>(?:/testbed/|\./)){re.escape(self.project_name)}/(?P<tail>[\w./-]+)",
            _replace_rooted_path,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"(?<![\w.-]){re.escape(self.project_name)}/(?P<tail>[\w./-]+)",
            lambda m: (
                self._record_path_notebook(self.virtual_project_name, self.project_name)
                or f"{self.virtual_project_name}/{_rewrite_segments(m.group('tail'), '/')}"
            ),
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"(?<![\w.-]){re.escape(self.project_name)}\.(?P<tail>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
            lambda m: (
                self._record_path_notebook(self.virtual_project_name, self.project_name)
                or f"{self.virtual_project_name}.{_rewrite_segments(m.group('tail'), '.')}"
            ),
            text,
            flags=re.IGNORECASE,
        )
        return text

    def to_real(self, text: str) -> str:
        """Translate virtual -> real.
        Uses session_notebook first (preferred), falls back to backward_map."""
        if not isinstance(text, str) or not text:
            return text
        return self._to_real_core(text)

    def to_real_command(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return self._rewrite_command_preserving_head(text, self._to_real_core)

    def to_real_patch(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return self._to_real_core(text)

    def _to_real_core(self, text: str) -> str:
        original_text = text

        # For method-aligned bundle mappings, reverse symbol translation is constrained
        # to names actually exposed during the session. Legacy flat mappings keep the
        # historical fallback behavior for compatibility.
        combined = {} if self.strict_session_reverse else dict(self._bwd_id_lookup or {})
        combined.update(self.session_notebook)

        if not combined and not self._reverse_namespace_path_rules:
            return text

        pattern, lookup = self._compile_exact_pattern(combined)
        path_rules = self.session_path_notebook if self.strict_session_reverse else self._reverse_namespace_path_rules
        path_rules_folded = {} if self.strict_session_reverse else self._reverse_namespace_path_rules_folded

        text = self._rewrite_paths_and_symbols(
            text,
            path_rules=path_rules,
            path_rules_folded=path_rules_folded,
            symbol_pattern=pattern,
            symbol_lookup=lookup,
            record_notebook=False,
            folded_symbols=False,
        )

        self.translate_count += 1
        self._log_translation(original_text, text, to_real=True)
        return text

    def _rewrite_command_preserving_head(self, text: str, rewrite_payload):
        if not isinstance(text, str) or not text:
            return text

        segments = _COMMAND_SPLIT_PATTERN.split(text)
        rewritten: list[str] = []

        for segment in segments:
            if not segment:
                continue
            if _COMMAND_SPLIT_PATTERN.fullmatch(segment):
                rewritten.append(segment)
                continue

            match = _COMMAND_PART_PATTERN.match(segment)
            if not match:
                rewritten.append(rewrite_payload(segment))
                continue

            prefix, head, tail = match.groups()
            if not head:
                rewritten.append(rewrite_payload(segment))
                continue

            # Environment assignment chains (e.g. FOO=1 BAR=2 cmd ...) should not be treated as command heads.
            if _ENV_ASSIGN_PATTERN.match(head):
                rewritten.append(rewrite_payload(segment))
                continue

            rewritten_tail = self._rewrite_python_c_payload(tail, rewrite_payload) if tail else tail
            rewritten.append(f"{prefix}{head}{rewritten_tail}")

        return "".join(rewritten)

    def _rewrite_python_c_payload(self, text: str, rewrite_payload):
        if not text or " -c " not in text:
            return rewrite_payload(text)

        def _replace(match):
            code = match.group("code")
            rewritten = rewrite_payload(code)
            return f"{match.group('prefix')}{match.group('quote')}{rewritten}{match.group('quote')}"

        rewritten = _PYTHON_C_PATTERN.sub(_replace, text)
        if rewritten == text:
            return rewrite_payload(text)
        return rewritten

    def _log_translation(self, original: str, result: str, *, to_real: bool):
        """Write a debug log entry."""
        try:
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                entry = {
                    "direction": "Virtual -> Real" if to_real else "Real -> Virtual",
                    "method": "table",
                    "input_len": len(original),
                    "output_len": len(result),
                    "changed": result != original,
                    "notebook_size": len(self.session_notebook),
                    "timestamp": time.time(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_translator_stats(self) -> dict:
        """Return translation statistics (table replacement, zero LLM cost)."""
        return {
            "translator_extra_cost": 0.0,
            "translator_extra_calls": 0,
            "translator_table_replacements": self.translate_count,
            "session_notebook_size": len(self.session_notebook),
            "session_path_notebook_size": len(self.session_path_notebook),
        }
