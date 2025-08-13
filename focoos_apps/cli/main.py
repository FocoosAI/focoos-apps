#!/usr/bin/env python3
"""
CLI for Focoos Apps.

This module provides a command-line interface for launching different Focoos applications.
Currently supports:
- smart-parking: Smart parking occupancy detection
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import available apps
from focoos_apps.apps.smart_parking import SmartParkingApp

# Initialize Typer app for CLI interface
app = typer.Typer(
    name="focoos-apps",
    help="Focoos AI Applications CLI",
    add_completion=False,
    rich_markup_mode="rich"
)

# Initialize console for rich output formatting
console = Console()


@app.command()
def list():
    """List all available Focoos applications."""
    table = Table(title="Available Focoos Applications")
    table.add_column("App Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")
    
    table.add_row(
        "smart-parking",
        "Smart parking occupancy detection with video/image processing",
        "✅ Available"
    )
    
    console.print(table)


@app.command()
def smart_parking(
    input_video: Optional[Path] = typer.Option(
        None, "--input-video", "-i", help="Input video file path"
    ),
    output_video: Optional[Path] = typer.Option(
        None, "--output-video", "-o", help="Output video file path"
    ),
    zones_file: Optional[Path] = typer.Option(
        None, "--zones-file", "-z", help="Zones configuration file path"
    ),
    api_key: str = typer.Option(
        ..., "--api-key", "-k", help="Focoos API key"
    ),
    model_ref: str = typer.Option(
        ..., "--model-ref", "-m", help="Model reference"
    ),
    runtime: str = typer.Option(
        "tensorrt", "--runtime", "-r", help="Runtime type (cpu, cuda, tensorrt)"
    )
):
    """
    Launch Smart Parking application.
    
    This application detects parking occupancy in videos using Focoos AI models.
    """
    try:
        # Validate inputs
        if not input_video:
            console.print("[red]Error: --input-video must be provided[/red]")
            raise typer.Exit(1)
        
        if input_video and not input_video.exists():
            console.print(f"[red]Error: Input video file not found: {input_video}[/red]")
            raise typer.Exit(1)

        # Create app instance
        app_config = {
            "api_key": api_key,
            "model_ref": model_ref,
            "runtime": runtime,
        }
        
        if zones_file:
            app_config["zones_file"] = str(zones_file)
        
        if input_video:
            app_config["input_video"] = str(input_video)
            if output_video:
                app_config["output_video"] = str(output_video)
        
        console.print(Panel.fit(
            "[bold cyan]Starting Smart Parking Application[/bold cyan]\n"
            f"Model: {model_ref}\n"
            f"Runtime: {runtime}",
            border_style="cyan"
        ))
        
        parking_app = SmartParkingApp(**app_config)
        
        # Process video
        console.print(f"[green]Processing video: {input_video}[/green]")
        parking_app.run()
        console.print("[green]Video processing completed![/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from focoos_apps import __version__
    console.print(f"[cyan]Focoos Apps CLI Version: {__version__}[/cyan]")


def app_cli():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    app()
