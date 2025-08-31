# Super Alita GUI Library

A lightweight, server-rendered GUI component system for quickly assembling simple dashboards and panels without heavy frontend frameworks.

## Goals

- Minimal dependencies (pure FastAPI + vanilla JS)
- Declarative Python component registration
- Progressive enhancement (HTML works without JS)
- Extensible styling via utility classes

## Structure

```text
src/gui/
  __init__.py
  registry.py        # component registry + built-ins
  router.py          # FastAPI routes (/gui, /gui/components/...)
```

## Usage

Register a component:

```python
from gui import register_component

@register_component("hello_box")
def hello_box(props: dict[str, str]) -> str:
    name = props.get("name", "World")
    return (
        "<div class='panel'><div class='panel-header'>Hello</div>"
        f"<div class='panel-body'>Hi {name}!</div></div>"
    )
```

Access via browser:

```text
/gui/components/hello_box?props={"name":"Alice"}
```

## Extending

- Add new components via decorators.
- Provide higher-order components that compose existing parts.
- Create client-side hydration by fetching JSON and replacing inner HTML.

## Future Enhancements

- Theming API
- State serialization snapshots
- Component composition DSL
- Streaming partial updates via SSE

## Testing

- `tests/test_gui_library.py` ensures core endpoints respond and built-ins render.

## License

Same as project root.
