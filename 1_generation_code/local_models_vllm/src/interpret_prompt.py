"""Prompt pieces from ../prompt.py, without its import-time side effects.

prompt.py cannot be imported (it loads .env, builds an OpenAI client, exits without
OPENAI_API_KEY and creates output directories at import), so the parts the local run
needs are copied here verbatim: the fallback system prompt, the JSON output
instruction and schema, extract_json, validate_response, the user-prompt assembly and
the SFT output shape. Keep them in sync with prompt.py if it changes.
"""
import json
import re

FALLBACK_SYSTEM_PROMPT = (
    "You are an expert computational biologist tasked with interpreting decision tree classification rules "
    "that predict specific biomarkers based on RNA gene expression and gene attributes. Your objective is "
    "to translate the mathematical logic of the tree, including feature hierarchy, split thresholds, and "
    "conditional dependencies, into a cohesive biological narrative that explains the underlying regulatory "
    "mechanisms. Using the specific biological context provided, you must synthesize these findings into "
    "concrete, experimentally testable hypotheses."
)

JSON_OUTPUT_INSTRUCTION = (
    "\n\nOUTPUT INSTRUCTION: Respond with a JSON object with exactly two string fields: "
    "\"thinking\" (your complete, step-by-step internal biological reasoning) and "
    "\"answer\" (your final biological hypothesis and narrative, written for the reader -- "
    "this is the only part they will see)."
)

# The `schema` part of prompt.py's RESPONSE_JSON_SCHEMA, for vLLM structured outputs.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["thinking", "answer"],
    "additionalProperties": False,
}


def extract_json(text):
    """First `{` to last `}`, or a ```json fenced block."""
    try:
        code_block = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block:
            return json.loads(code_block.group(1))

        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            json_str = text[start:end + 1]
            return json.loads(json_str)

        return json.loads(text)
    except Exception:
        return None


_TRAILING_COMMA = re.compile(r',(\s*[}\]])')
_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def extract_json_lenient(text):
    """extract_json, then the syntax slips Qwen3.5 makes in long JSON strings (not in prompt.py):
    a trailing comma before } or ], markdown escapes such as \\_ that JSON rejects, and raw
    control characters inside strings. Structure and fields are not repaired."""
    obj = extract_json(text)
    if obj is not None:
        return obj
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1:
        return None
    body = _INVALID_ESCAPE.sub('', _TRAILING_COMMA.sub(r'\1', text[start:end + 1]))
    try:
        return json.loads(body, strict=False)
    except Exception:
        return None


def validate_response(obj):
    if not isinstance(obj, dict):
        return False, "response was not a JSON object"
    thinking = obj.get("thinking")
    answer = obj.get("answer")
    if not thinking or not isinstance(thinking, str) or len(thinking.strip()) < 20:
        return False, "missing or suspiciously short 'thinking' field"
    if not answer or not isinstance(answer, str) or len(answer.strip()) < 50:
        return False, "missing or suspiciously short 'answer' field"
    return True, "ok"


def build_prompts(data):
    """(system, user) for one input row, as prompt.py builds them (before the JSON instruction)."""
    system = data.get('system_prompt') or FALLBACK_SYSTEM_PROMPT
    user = data.get('user_prompt') or (
        f"{data.get('biological_context', '')}\n\n"
        f"{data.get('focused_zones_text', '')}\n\n"
        f"{data.get('raw_tree_text', '')}\n\n"
        f"{data.get('task_text', '')}"
    ).strip()
    return system, user


def sft_row(key, system, user, thinking, answer):
    content = f"<think>\n{thinking.strip()}\n</think>\n<answer>\n{answer.strip()}\n</answer>"
    return {
        "key": key,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": content},
        ],
    }
