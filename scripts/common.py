# scripts/common.py
"""
Common utilities and abstractions for the Spec Kit Python toolchain.
Provides shared classes for project management, AI integration,
and Git operations.
"""

import os
import subprocess
from pathlib import Path

import openai
from git import Repo


# --- AI Client Abstraction ---
def get_ai_client() -> openai.OpenAI:
    """Initializes and returns the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return openai.OpenAI(api_key=api_key)


def invoke_ai_generation(
    prompt: str, system_prompt: str, model: str = "gpt-4-turbo"
) -> str:
    """
    Sends a prompt to the configured AI model and returns the response.

    Args:
        prompt: The user prompt to send to the AI
        system_prompt: The system prompt defining AI behavior
        model: The OpenAI model to use

    Returns:
        The AI-generated response content
    """
    client = get_ai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    return content if content is not None else ""


# --- Project Context Class ---
class Project:
    """
    A single, reliable source for project paths and state.
    Centralizes all project-related path management and common operations.
    """

    def __init__(self, root: Path = Path(".")):
        self.root = root.resolve()
        self.memory_dir = self.root / "memory"
        self.specs_dir = self.root / "specs"
        self.templates_dir = self.root / "templates"
        self.scripts_dir = self.root / "scripts"

    def get_constitution(self) -> str:
        """Reads and caches the project constitution."""
        path = self.memory_dir / "constitution.md"
        return path.read_text() if path.exists() else ""

    def get_feature_path(self, feature_name: str) -> Path | None:
        """Finds the directory for a given feature."""
        # Search for directories matching the feature name pattern
        slug = feature_name.lower().replace(" ", "-").replace("_", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")

        for path in self.specs_dir.glob("???-*"):
            if slug in path.name:
                return path
        return None

    def create_feature_directory(self, feature_name: str) -> Path:
        """Create a numbered feature directory."""
        # Find next available number
        existing_features = list(self.specs_dir.glob("???-*"))
        next_num = len(existing_features) + 1

        # Create slug from feature name
        slug = feature_name.lower().replace(" ", "-").replace("_", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")

        feature_path = self.specs_dir / f"{next_num:03d}-{slug}"
        feature_path.mkdir(parents=True, exist_ok=True)
        return feature_path


# --- Git Integration ---
def create_feature_branch(branch_name: str) -> None:
    """Creates and checks out a new git branch."""
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)


def commit_changes(message: str, add_path: str) -> None:
    """Adds and commits files to the current branch."""
    subprocess.run(["git", "add", add_path], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)


def get_repo() -> Repo:
    """Get the git repository object."""
    return Repo(".")


# --- Template Utilities ---
def load_template(project: Project, template_name: str) -> str:
    """Load a Jinja2 template from the project's templates directory."""
    template_path = project.templates_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template {template_name} not found in {project.templates_dir}"
        )
    return template_path.read_text()
