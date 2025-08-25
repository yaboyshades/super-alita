import hashlib
import uuid

# Fixed namespace for tool IDs (create once and do not change afterward)
# If you already have a project-wide namespace, you can replace this with it.
NAMESPACE_TOOLS = uuid.UUID("5b2c7a51-3121-4a58-b1cf-0a39b6b2e3b8")


def _content_key(s: str, threshold: int = 160) -> str:
    s = (s or "").strip()
    if len(s) <= threshold:
        return s
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_tool_id(
    title: str, code: str, tool_type: str = "TOOL", extra: str | None = None
) -> str:
    """
    UUIDv5(tool_namespace, f"{tool_type}|{title}|{content_key}|{extra}")
    ensures same (title, code) => same id.
    """
    name = "|".join(
        [
            tool_type,
            (title or "").strip(),
            _content_key(code or ""),
            (extra or "").strip(),
        ]
    )
    return str(uuid.uuid5(NAMESPACE_TOOLS, name))
