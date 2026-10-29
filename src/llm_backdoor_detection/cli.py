from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .families import FAMILIES
from .motif import plan_motif, run_motif
from .specguard import status as specguard_status
from .upstream import UpstreamNotFoundError, find_bdrscan_root

app = typer.Typer(help="LLM backdoor detection orchestration CLI.")
motif_app = typer.Typer(help="Run Microsoft MOTIF workflows.")
specguard_app = typer.Typer(help="Report SpecGuard support status.")
app.add_typer(motif_app, name="motif")
app.add_typer(specguard_app, name="specguard")
console = Console()


@app.command("capabilities")
def capabilities() -> None:
    """Show supported local and upstream-backed workflows."""
    table = Table(title="LLM Backdoor Detection Capabilities")
    table.add_column("Area")
    table.add_column("Status")
    table.add_column("Notes")
    try:
        root = find_bdrscan_root()
        motif_status = "available"
        motif_notes = str(root)
    except UpstreamNotFoundError as exc:
        motif_status = "needs upstream checkout"
        motif_notes = str(exc)
    table.add_row("MOTIF / llm-backdoor-scanner", motif_status, motif_notes)
    sg = specguard_status()
    table.add_row("SpecGuard", "unsupported" if not sg.implemented else "available", sg.reason)
    table.add_row("Families", "available", ", ".join(sorted(FAMILIES)))
    console.print(table)


@app.command("validate-config")
def validate_config(config: Path) -> None:
    """Validate a detection YAML file."""
    loaded = load_config(config)
    console.print(f"[green]Valid config:[/green] {loaded.name}")
    console.print(f"Family: {loaded.family}")
    console.print(f"Target model: {loaded.target_model}")
    if loaded.base_model:
        console.print(f"Base model: {loaded.base_model}")


@motif_app.command("plan")
def motif_plan(config: Path, stage: str | None = typer.Option(None, help="Run only one stage.")) -> None:
    """Print the Microsoft MOTIF commands that would run."""
    loaded = load_config(config)
    commands = plan_motif(loaded, config, selected_stage=stage)
    for command in commands:
        console.print(f"[bold]{command.stage}[/bold]")
        console.print(command.shell_display())


@motif_app.command("run")
def motif_run(config: Path, stage: str | None = typer.Option(None, help="Run only one stage.")) -> None:
    """Run Microsoft MOTIF commands."""
    loaded = load_config(config)
    try:
        statuses = run_motif(loaded, config, selected_stage=stage)
    except FileNotFoundError as exc:
        console.print(f"Cannot start MOTIF: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"[green]MOTIF completed[/green] stages={len(statuses)}")


@specguard_app.command("status")
def specguard_status_cmd() -> None:
    """Show SpecGuard implementation status."""
    sg = specguard_status()
    console.print("[yellow]SpecGuard is not implemented in this project.[/yellow]")
    console.print(sg.reason)
    console.print(sg.paper_summary)


if __name__ == "__main__":
    app()

