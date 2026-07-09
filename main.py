"""Vision Workbench — entry point.

Usage:
    python main.py                          → run full pipeline with default config
    python main.py configs/detect.yaml      → run with custom config
    python main.py --dry-run                → preview execution plan
"""

from run_pipeline import main

if __name__ == "__main__":
    import sys
    config = next((a for a in sys.argv[1:] if not a.startswith("--")), "configs/detect.yaml")
    dry_run = "--dry-run" in sys.argv
    main(config, dry_run)
