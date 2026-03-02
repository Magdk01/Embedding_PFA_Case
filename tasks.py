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
def serve(c, host="127.0.0.1", port=8001, reload=False):
    """Start the API server."""
    reload_flag = " --reload" if reload else ""
    c.run(f"uv run uvicorn embedding_pfa_case.api:app --host {host} --port {port}{reload_flag}")


@task
def run_model(c):
    """Run the model."""
    c.run("uv run python -m embedding_pfa_case.model")


DOCKER_IMAGE = "embedding-api"
DOCKER_FILE = "dockerfiles/api.dockerfile"


@task
def docker_build(c, tag=DOCKER_IMAGE):
    """Build the Docker image."""
    c.run(f"docker build -f {DOCKER_FILE} -t {tag} .")


@task
def docker_run(c, tag=DOCKER_IMAGE, port=8000):
    """Run the API in a Docker container."""
    c.run(f"docker run --rm -p {port}:8000 {tag}")


@task(format_check, lint_check, typecheck, test)
def check(c):
    """Run format-check, lint-check, typecheck, and tests."""
    print("All checks passed.")
