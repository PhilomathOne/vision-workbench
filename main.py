"""Vision Workbench entry point — delegates to the CLI.

Usage:
    python main.py             → Equivalent to `vw --help`
    python main.py run ...     → Equivalent to `vw run ...`
"""

from vision_workbench.cli.app import app

if __name__ == "__main__":
    app()
