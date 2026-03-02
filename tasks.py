"""Invoke tasks for this project."""

from invoke import task


@task
def install(c):
    """Install dependencies with uv."""
    c.run("uv sync --locked --dev")


@task
def test(c):
    """Run pytest."""
    c.run("uv run pytest tests/")


@task
def test_cov(c):
    """Run pytest with coverage."""
    c.run("uv run coverage run -m pytest tests/")
    c.run("uv run coverage report -m")


@task
def format(c):
    """Format code with ruff."""
    c.run("uv run ruff format .")


@task
def format_check(c):
    """Check code formatting only."""
    c.run("uv run ruff format . --check")


@task
def lint(c):
    """Lint and fix with ruff."""
    c.run("uv run ruff check . --fix")


@task
def lint_check(c):
    """Lint only, no fixes."""
    c.run("uv run ruff check .")


@task
def typecheck(c):
    """Run mypy."""
    c.run("uv run mypy .")



@task
def run_model(c):
    """Run the model."""
    c.run("uv run python -m embedding_pfa_case.model")

@task(format_check, lint_check, typecheck, test)
def check(c):
    """Run format-check, lint-check, typecheck, and tests."""
    print("All checks passed.")
