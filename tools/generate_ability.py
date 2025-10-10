#!/usr/bin/env python3
"""
Constitutional Ability Generator

Automates creation of new abilities following Super-Alita's
constitutional requirements with test-first methodology.

Usage examples:
  python tools/generate_ability.py --name "data_processor" --description "Processes data files"
  python tools/generate_ability.py --name "web_analyzer" --author "Your Name" --mode test-only
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


def to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).replace(" ", "_").lower()


def to_pascal_case(name: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[_\s]+", name))


def to_title_case(name: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[_\s]+", name))


def validate_ability_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or not re.match(r"^[a-zA-Z]", clean):
        raise ValueError(
            "Ability name must start with a letter and be non-empty"
        )
    return to_snake_case(clean)


def substitute(content: str, vars: dict[str, str]) -> str:
    for k, v in vars.items():
        content = content.replace(f"___{k}___", v)
    return content


def create_from_template(
    template_file: Path, output_file: Path, variables: dict[str, str]
) -> Path:
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    text = template_file.read_text(encoding="utf-8")
    text = substitute(text, variables)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    print(f"✅ Created: {output_file}")
    return output_file


def update_abilities_init(
    ability_name: str, class_name: str, abilities_dir: Path
) -> None:
    init_file = abilities_dir / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
    else:
        content = '"""Super-Alita Abilities Package"""\n\n'

    import_line = f"from .{ability_name}_ability import {class_name}Ability"
    if import_line not in content:
        if "from ." in content:
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("from ."):
                    insert_at = i + 1
            lines.insert(insert_at, import_line)
            content = "\n".join(lines)
        else:
            content += f"\n{import_line}\n"

    if "__all__" in content:
        m = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if m:
            inner = m.group(1)
            entry = f'"{class_name}Ability"'
            if entry not in inner:
                new_inner = (
                    (inner.rstrip() + f",\n    {entry}")
                    if inner.strip()
                    else f"\n    {entry}\n"
                )
                content = content.replace(
                    m.group(0), f"__all__ = [{new_inner}]"
                )
    else:
        content += f'\n\n__all__ = [\n    "{class_name}Ability"\n]\n'

    init_file.write_text(content, encoding="utf-8")
    print(f"✅ Updated: {init_file}")


def generate_registration_instructions(
    ability_name: str, class_name: str
) -> None:
    print(
        f"""
📋 ABILITY REGISTRATION INSTRUCTIONS

1) Import in src/main.py:
   from src.abilities.{ability_name}_ability import {class_name}Ability

2) Register in ability registry (look for SimpleAbilityRegistry usage):
   try:
       obj = {class_name}Ability()
       await obj.initialize(event_bus)
       abilities["{ability_name}"] = obj
       print("✅ Registered ability: {ability_name}")
   except Exception as e:
       print(f"❌ Failed to register {ability_name}: {{e}}")

3) Expose API endpoint if needed in FastAPI router:
   @app.post("/ability/execute/{ability_name}")
   async def execute_{ability_name}(request: Dict[str, Any]):
       ability = ability_registry.get("{ability_name}")
       if not ability:
           raise HTTPException(status_code=404, detail="Ability not found")
       return await ability.execute(request)

4) Run tests:
   pytest tests/abilities/test_{ability_name}_ability.py -q
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Super-Alita abilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/generate_ability.py --name text_processor --description 'Processes text'\n"
            "  python tools/generate_ability.py --name web_scraper --mode test-only\n"
        ),
    )
    parser.add_argument(
        "--name", required=True, help="Ability name (snake_case or CamelCase)"
    )
    parser.add_argument(
        "--description",
        default="A new Super-Alita ability",
        help="Ability description",
    )
    parser.add_argument(
        "--author", default="Super-Alita Developer", help="Author name"
    )
    parser.add_argument("--version", default="1.0.0", help="Initial version")
    parser.add_argument(
        "--input-field", default="input_data", help="Primary input field name"
    )
    parser.add_argument(
        "--mode",
        choices=["both", "test-only", "implementation-only"],
        default="both",
        help="Generation mode",
    )
    parser.add_argument(
        "--output-dir", help="Output base directory (defaults to project root)"
    )

    args = parser.parse_args()

    ability_name = validate_ability_name(args.name)
    class_name = to_pascal_case(ability_name)
    variables: dict[str, str] = {
        "abilityName": ability_name,
        "AbilityName": class_name,
        "abilityDescription": args.description,
        "author": args.author,
        "version": args.version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "inputField": args.input_field,
        "inputDescription": f"Input data for {to_title_case(ability_name)}",
    }

    base_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent.parent
    )
    template_dir = base_dir / "templates"
    abilities_dir = base_dir / "src" / "abilities"
    tests_dir = base_dir / "tests" / "abilities"

    print(f"🚀 Generating ability: {ability_name}")
    print(f"   Class name: {class_name}Ability")
    print(f"   Mode: {args.mode}")

    if args.mode in ("both", "test-only"):
        test_tmpl = (
            template_dir
            / "ability_test_template"
            / "test____abilityName____ability.py"
        )
        test_out = tests_dir / f"test_{ability_name}_ability.py"
        create_from_template(test_tmpl, test_out, variables)

    if args.mode in ("both", "implementation-only"):
        impl_tmpl = (
            template_dir
            / "ability_implementation_template"
            / "___abilityName____ability.py"
        )
        impl_out = abilities_dir / f"{ability_name}_ability.py"
        create_from_template(impl_tmpl, impl_out, variables)
        update_abilities_init(ability_name, class_name, abilities_dir)

    generate_registration_instructions(ability_name, class_name)
    print("\n✅ Generation complete.")
    if args.mode in ("both", "test-only"):
        print(
            f"Next: pytest tests/abilities/test_{ability_name}_ability.py -q (expected to fail first)"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"❌ Error: {e}")
        sys.exit(1)
