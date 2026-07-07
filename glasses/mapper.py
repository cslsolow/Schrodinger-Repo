import json
import keyword
import random
import logging
import concurrent.futures
import re
import time
from typing import Dict, List, Any, Set
from rich.progress import Progress
import litellm
from litellm import cost_calculator

litellm.suppress_debug_info = True

from minisweagent.models import GLOBAL_MODEL_STATS, get_model

logger = logging.getLogger(__name__)

TOKEN_BATCH_MAX_ATTEMPTS = 4

_IDENTIFIER_FRAGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")


class SemanticMapper:
    def __init__(self, model_name: str = None, model_config: Dict = None, require_model: bool = True):
        self.model = get_model(model_name, model_config) if require_model else None
        self.token_candidates_cache: Dict[str, List[str]] = {}

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        text = (content or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
        return text.strip()

    def _completion_cost(self, response) -> float:
        model_name = self.model.config.model_name
        cost_tracking = getattr(self.model.config, "cost_tracking", "ignore_errors")
        try:
            cost = float(cost_calculator.completion_cost(response, model=model_name))
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
            return cost
        except Exception as e:
            if cost_tracking == "ignore_errors":
                logger.debug("completion_cost skipped for %s: %s", model_name, e)
            else:
                logger.warning(
                    "completion_cost failed for %s (%s); cost recorded as 0. "
                    "Set cost_tracking: ignore_errors on the model config to silence.",
                    model_name,
                    e,
                )
            return 0.0

    @staticmethod
    def _model_requires_temperature_one(model_name: str) -> bool:
        lowered = model_name.lower()
        return "gpt-5" in lowered

    @classmethod
    def _is_valid_python_identifier_fragment(cls, token: str) -> bool:
        if not token or not _IDENTIFIER_FRAGMENT_RE.match(token):
            return False
        if keyword.iskeyword(token.lower()):
            return False
        return True

    def _normalize_token_batch_dict(self, data: dict, batch: List[str]) -> Dict[str, List[str]]:
        lowered_key_map: Dict[str, Any] = {}
        for raw_key, raw_val in data.items():
            lowered_key_map[str(raw_key).strip().lower()] = raw_val

        out: Dict[str, List[str]] = {}
        for word in batch:
            raw_list = lowered_key_map.get(word)
            if not isinstance(raw_list, list):
                raw_list = []

            seen: Set[str] = set()
            cleaned: List[str] = []
            for item in raw_list:
                if not isinstance(item, str):
                    continue
                frag = item.strip()
                if not self._is_valid_python_identifier_fragment(frag):
                    continue
                low = frag.lower()
                if low == word or low in seen:
                    continue
                seen.add(low)
                cleaned.append(low)
                if len(cleaned) >= 5:
                    break

            out[word] = cleaned if cleaned else [word]
        return out

    def _process_token_batch(self, batch: List[str], context: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
        context_str = ""
        if context:
            context_str = "\nCONTEXTUAL IDENTIFIER FAMILIES (to help you understand usage):\n"
            for token, examples in context.items():
                if examples:
                    relevant_examples = [ex for ex in examples if token in ex.lower()][:5]
                    if relevant_examples:
                        context_str += f"- {token}: used in {', '.join(relevant_examples)}\n"

        prompt = f"""You are a linguistic expert specializing in computer science terminology.
For each of the following programming identifiers or tokens, provide 5 semantically equivalent but lexically different alternative terms.

CRITICAL CONSTRAINTS:
1. LEXICAL DIVERSITY: Strictly avoid using the original term or its direct derivatives.
2. CONTEXTUAL CONSISTENCY: Some tokens belong to the same "family" (e.g., 'datetime' prefix). Ensure the mappings you provide would form natural, professional-sounding identifiers when combined.
3. SUFFIX DIFFERENTIATION: For tokens that often appear as suffixes in a family, ensure their alternatives remain distinct and descriptive.
4. NO BRAND NAMES: Do not use names like 'django', 'flask', etc.
5. VALID PYTHON IDENTIFIER FRAGMENTS: Every alternative MUST be a single ASCII token that is a valid Python identifier (letters, digits, underscore only; MUST NOT start with a digit). NO spaces, hyphens, dots, slashes, or punctuation. Use lowercase_snake style fragments like "user_prefs" not "user prefs".
6. NO PYTHON KEYWORDS: Do not output reserved words (e.g. class, def, return, import, True, False, None).
{context_str}
Return ONLY a JSON object: {{"original_term": ["alt1", "alt2", "alt3", "alt4", "alt5"]}}

Terms:
{json.dumps(batch)}
"""
        model_name = self.model.config.model_name
        fixed_temp_one = self._model_requires_temperature_one(model_name)
        last_error: Exception | None = None
        for attempt in range(TOKEN_BATCH_MAX_ATTEMPTS):
            call_kwargs = dict(self.model.config.model_kwargs)
            if fixed_temp_one:
                call_kwargs["temperature"] = 1.0
            elif attempt == 0:
                call_kwargs.setdefault("temperature", 1.0)
            else:
                call_kwargs["temperature"] = min(0.55, 0.25 + 0.15 * attempt)

            user_content = prompt
            if attempt > 0 and fixed_temp_one:
                user_content += (
                    "\n\nYour previous reply was not valid JSON. Reply with exactly one JSON object, "
                    "double-quoted keys and strings only, no trailing commas, no markdown fences. "
                    "Each alternative string must be a valid Python identifier fragment (ASCII letters, digits, underscore only)."
                )

            try:
                raw = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": user_content}],
                    **call_kwargs,
                )
                GLOBAL_MODEL_STATS.add(self._completion_cost(raw))
                content = self._strip_code_fence(raw.choices[0].message.content or "")
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise TypeError(f"Expected JSON object, got {type(data).__name__}")
                return self._normalize_token_batch_dict(data, batch)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                last_error = e
                logger.warning(
                    "token batch JSON parse failed (attempt %s/%s): %s: %s",
                    attempt + 1,
                    TOKEN_BATCH_MAX_ATTEMPTS,
                    type(e).__name__,
                    e,
                )
                if attempt + 1 < TOKEN_BATCH_MAX_ATTEMPTS:
                    time.sleep(0.35 * (2**attempt))
            except Exception as e:
                last_error = e
                logger.error(
                    "Error in token batch (attempt %s/%s): %s: %s",
                    attempt + 1,
                    TOKEN_BATCH_MAX_ATTEMPTS,
                    type(e).__name__,
                    e,
                )
                if attempt + 1 < TOKEN_BATCH_MAX_ATTEMPTS:
                    time.sleep(0.35 * (2**attempt))

        logger.error(
            "token batch exhausted retries after %s attempts: %s",
            TOKEN_BATCH_MAX_ATTEMPTS,
            repr(last_error) if last_error else "unknown",
        )
        return {word: [word] for word in batch}

    def generate_token_candidates(self, tokens: List[str], brand: str = None, context: Dict[str, List[str]] = None, max_workers: int = 5) -> Dict[str, List[str]]:
        if self.model is None:
            raise ValueError("A model is required to generate token candidates.")

        tokens_to_query = []
        for t in tokens:
            t_key = t.lower()
            if t_key in self.token_candidates_cache:
                continue

            if brand and t_key == brand.lower():
                self.token_candidates_cache[t_key] = ["app", "core", "engine", "base", "hub"]
            else:
                tokens_to_query.append(t_key)

        if tokens_to_query:
            batch_size = 30
            batches = [tokens_to_query[i : i + batch_size] for i in range(0, len(tokens_to_query), batch_size)]
            with Progress() as progress:
                task = progress.add_task("[cyan]Generating token candidates...", total=len(batches))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {executor.submit(self._process_token_batch, b, context): b for b in batches}
                    failures = []
                    for future in concurrent.futures.as_completed(future_to_batch):
                        try:
                            self.token_candidates_cache.update(future.result())
                        except Exception as exc:
                            failures.append(exc)
                        n_calls = getattr(self.model, "n_calls", GLOBAL_MODEL_STATS.n_calls)
                        cost = getattr(self.model, "cost", GLOBAL_MODEL_STATS.cost)
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Mapping: {n_calls} calls, Cost: ${cost:.4f}",
                        )
            if failures:
                raise RuntimeError(f"Failed to generate token candidates for {len(failures)} batch(es)") from failures[0]

            missing_tokens = [t for t in tokens_to_query if t not in self.token_candidates_cache]
            if missing_tokens:
                raise RuntimeError(f"Missing token candidates after generation: {missing_tokens[:10]!r}")

        return self.token_candidates_cache

    def create_token_mapping(self, tokens: List[str], seed: int, avoid_tokens: Set[str] = None) -> Dict[str, str]:
        rng = random.Random(seed)
        mapping = {}
        used_virtual_tokens = set()
        avoid_tokens = avoid_tokens or set()

        for t_key in sorted(list(set(t.lower() for t in tokens))):
            candidates = self.token_candidates_cache.get(t_key, [t_key])
            shuffled = list(candidates)
            rng.shuffle(shuffled)

            selected = t_key
            for c in shuffled:
                c_low = c.lower()
                if c_low not in used_virtual_tokens and c_low != t_key and c_low not in avoid_tokens:
                    selected = c_low
                    break
            mapping[t_key] = selected
            used_virtual_tokens.add(selected)
        return mapping

    def reconstruct_identifier(self, original: str, token_mapping: Dict[str, str], extractor) -> str:
        suffix = ""
        name = original
        if "." in original:
            name, suffix = original.rsplit(".", 1)
            suffix = "." + suffix

        orig_tokens = extractor.tokenize_identifier(name)
        new_parts = []

        for ot in orig_tokens:
            vt = token_mapping.get(ot.lower(), ot.lower())

            if ot.isupper() and len(ot) > 1:
                new_parts.append(vt.upper())
            elif ot[0].isupper():
                new_parts.append(vt.capitalize())
            else:
                new_parts.append(vt.lower())

        if "_" in original:
            reconstructed = "_".join(new_parts)
        else:
            reconstructed = "".join(new_parts)
            if name and name[0].islower() and reconstructed:
                reconstructed = reconstructed[0].lower() + reconstructed[1:]

        return reconstructed + suffix

    def reconstruct_identifier_with_offset(self, original: str, token_mapping: Dict[str, str], extractor, seed_offset: int = 0) -> str:
        if seed_offset == 0:
            return self.reconstruct_identifier(original, token_mapping, extractor)

        offset_mapping = {}
        for token, virtual in token_mapping.items():
            candidates = self.token_candidates_cache.get(token, [token])
            if len(candidates) > 1:
                try:
                    idx = [c.lower() for c in candidates].index(virtual.lower())
                except ValueError:
                    idx = 0
                new_idx = (idx + seed_offset) % len(candidates)
                offset_mapping[token] = candidates[new_idx].lower()
            else:
                offset_mapping[token] = virtual
        return self.reconstruct_identifier(original, offset_mapping, extractor)
