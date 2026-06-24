"""Vision Workbench CLI — entry point for all commands.

Usage:
    vw --help              Show available commands
    vw run config.yaml     Execute a pipeline
    vw list detectors      List registered detectors
    vw serve               Start web UI
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from vision_workbench import __version__

app = typer.Typer(
    name="vw",
    help="Vision Workbench — End-to-end Computer Vision MLOps Platform",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit"),
):
    """Vision Workbench — End-to-end CV MLOps Platform."""
    if version:
        console.print(f"[bold]Vision Workbench[/] v{__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# vw run — Execute pipeline
# ---------------------------------------------------------------------------

@app.command()
def run(
    config: Optional[Path] = typer.Argument(None, help="Path to pipeline YAML config"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Run only this stage"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from run ID"),
):
    """Execute a pipeline from a YAML configuration file."""
    if config is None:
        console.print("[red]Error:[/] No config file specified. Usage: vw run <config.yaml>")
        raise typer.Exit(code=1)

    if not config.exists():
        console.print(f"[red]Error:[/] Config file not found: {config}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Vision Workbench[/] v{__version__}")
    console.print(f"Pipeline config: {config}")

    try:
        from vision_workbench.core.config import PipelineConfig

        pipeline_config = PipelineConfig.from_yaml(str(config))
        console.print(f"[green]✓[/] Config loaded: {pipeline_config.name}")
        console.print(f"  Stages: {pipeline_config.stages or '(none specified)'}")

        if dry_run:
            console.print("[yellow]Dry run mode[/] — no execution")
            return

        console.print("[yellow]Pipeline runner not yet implemented[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# vw list — List registered components
# ---------------------------------------------------------------------------

@app.command()
def list(
    component: str = typer.Argument("detectors", help="What to list: detectors, formats, frameworks, platforms, tasks"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Filter by task type"),
):
    """List registered detectors, formats, frameworks, or platforms."""
    from vision_workbench.core.types import TaskType

    if component == "tasks":
        table = Table(title="Supported Task Types")
        table.add_column("Task", style="cyan")
        table.add_column("Description", style="green")
        for task_type in TaskType:
            table.add_row(task_type.value, _task_description(task_type.value))
        console.print(table)
        return

    if component == "detectors":
        from vision_workbench.core.registry import detector_registry

        table = Table(title="Registered Detectors")
        table.add_column("Name", style="cyan")
        table.add_column("Task", style="green")
        table.add_column("Framework", style="yellow")
        table.add_column("Description")

        entries = detector_registry.list()
        if task:
            entries = detector_registry.list_by(task=task)

        if not entries:
            console.print("[yellow]No detectors registered yet.[/]")
            console.print("Detectors are auto-registered when their modules are imported.")
            return

        for name, cls in entries.items():
            meta = getattr(cls, "_registry_metadata", {})
            table.add_row(
                name,
                meta.get("task", cls.task_type if hasattr(cls, "task_type") else "—"),
                meta.get("framework", cls.framework if hasattr(cls, "framework") else "—"),
                meta.get("description", "—"),
            )
        console.print(table)
        return

    if component == "formats":
        from vision_workbench.core.registry import format_registry

        table = Table(title="Annotation Format Converters")
        table.add_column("Name", style="cyan")
        table.add_column("Direction", style="green")
        entries = format_registry.list()
        if not entries:
            console.print("[yellow]No format converters registered yet.[/]")
            return
        for name, cls in entries.items():
            meta = getattr(cls, "_registry_metadata", {})
            table.add_row(name, meta.get("direction", "—"))
        console.print(table)
        return

    if component == "frameworks":
        from vision_workbench.core.registry import framework_registry

        table = Table(title="Training Framework Adapters")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        entries = framework_registry.list()
        if not entries:
            console.print("[yellow]No framework adapters registered yet.[/]")
            return
        for name, cls in entries.items():
            meta = getattr(cls, "_registry_metadata", {})
            table.add_row(name, meta.get("description", "—"))
        console.print(table)
        return

    if component == "platforms":
        from vision_workbench.core.registry import exporter_registry

        table = Table(title="Export Platforms")
        table.add_column("Platform", style="cyan")
        table.add_column("Description", style="green")
        entries = exporter_registry.list()
        if not entries:
            console.print("[yellow]No exporters registered yet.[/]")
            return
        for name, cls in entries.items():
            meta = getattr(cls, "_registry_metadata", {})
            table.add_row(name, meta.get("description", "—"))
        console.print(table)
        return

    console.print(f"[red]Unknown component: {component}[/]")
    console.print("Available: detectors, formats, frameworks, platforms, tasks")


# ---------------------------------------------------------------------------
# vw detect — Quick single inference
# ---------------------------------------------------------------------------

@app.command()
def detect(
    source: str = typer.Argument(..., help="Image or video path, or 'webcam'"),
    detector: str = typer.Option("yolo", "--detector", "-d", help="Detector name from registry"),
    show: bool = typer.Option(True, "--show/--no-show", help="Display results in window"),
    save: Optional[Path] = typer.Option(None, "--save", help="Save annotated output to directory"),
    output_format: str = typer.Option("image", "--format", "-f", help="Output format: image, json, video"),
):
    """Run quick inference with a registered detector."""
    console.print(f"[bold]Detection:[/] source={source}, detector={detector}")
    console.print("[yellow]Inference engine not yet implemented[/]")


# ---------------------------------------------------------------------------
# vw data — Dataset management
# ---------------------------------------------------------------------------

@app.command()
def data(
    action: str = typer.Argument("status", help="Action: status, clean, split, convert"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source path"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target path"),
):
    """Manage datasets: status, clean, split, convert annotations."""
    if action == "status" and source:
        src = Path(source)
        if src.exists():
            images = list(src.rglob("*.jpg")) + list(src.rglob("*.png"))
            console.print(f"[bold]Dataset Status:[/] {source}")
            console.print(f"  Total images: {len(images)}")
        else:
            console.print(f"[red]Path not found: {source}[/]")
    elif action == "clean" and source and target:
        console.print(f"[bold]Cleaning dataset:[/] {source} → {target}")
        console.print("[yellow]Use: vw run configs/data_prepare.yaml[/]")
    else:
        console.print("[yellow]Use: vw run <config.yaml> for full data pipeline[/]")
        console.print("  vw data status --source <path>")


# ---------------------------------------------------------------------------
# vw model — Model management
# ---------------------------------------------------------------------------

@app.command()
def model(
    action: str = typer.Argument("list", help="Action: list, pull, info, register"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Model name"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Checkpoint path"),
):
    """Manage models: list zoo, pull pre-trained, register checkpoints."""
    from vision_workbench.models.zoo import KNOWN_MODELS, ModelZoo

    zoo = ModelZoo()

    if action == "list":
        cached = zoo.list()
        known = list(KNOWN_MODELS.keys())
        table = Table(title="Model Zoo")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Info", style="yellow")
        for m in known:
            status = "[green]cached[/]" if m in cached else "[dim]not cached[/]"
            info = KNOWN_MODELS.get(m, {})
            table.add_row(m, status, f"{info.get('task', '—')}, {info.get('framework', '—')}")
        console.print(table)
        if not known:
            console.print("[yellow]No known models. Use vw model pull <name>[/]")

    elif action == "pull" and name:
        try:
            path = zoo.pull(name)
            console.print(f"[green]Downloaded:[/] {path}")
        except KeyError as e:
            console.print(f"[red]{e}[/]")

    elif action == "info" and name:
        info = zoo.info(name)
        if info:
            console.print(f"[bold]{name}[/]")
            for k, v in info.items():
                console.print(f"  {k}: {v}")
        else:
            console.print(f"[red]Unknown model: {name}[/]")

    elif action == "register" and checkpoint:
        console.print(f"[bold]Registering model:[/] {name or 'unnamed'}")
        console.print(f"  Checkpoint: {checkpoint}")
        console.print("[yellow]ModelRegistry not yet fully implemented[/]")
    else:
        console.print("[yellow]Usage: vw model list|pull|info|register[/]")


# ---------------------------------------------------------------------------
# vw serve — Start web UI or API server
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address"),
    port: int = typer.Option(7860, "--port", "-p", help="Bind port"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model checkpoint to serve"),
    api_only: bool = typer.Option(False, "--api", help="Launch API server only (no UI)"),
):
    """Start the Gradio web UI or API server."""
    if api_only:
        console.print(f"[bold]Starting Vision Workbench API[/] on {host}:{port}")
        import uvicorn
        uvicorn.run("vision_workbench.serve.app:app", host=host, port=port, reload=False)
    else:
        from vision_workbench.serve.gradio_ui import launch
        console.print(f"[bold]Starting Vision Workbench UI[/] on http://{host}:{port}")
        launch(host=host, port=port)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_description(task: str) -> str:
    """Return a human-readable description for a task type."""
    descriptions = {
        "object_detection": "Detect and locate objects with bounding boxes",
        "classification": "Classify an entire image into categories",
        "instance_segmentation": "Segment individual object instances",
        "semantic_segmentation": "Classify each pixel into categories",
        "face_detection": "Detect human faces in images",
        "pose_estimation": "Estimate human body keypoints/skeleton",
        "hand_tracking": "Track hand landmarks and gestures",
        "ocr": "Recognize text in images",
        "feature_matching": "Match visual features between images",
    }
    return descriptions.get(task, "—")


if __name__ == "__main__":
    app()
