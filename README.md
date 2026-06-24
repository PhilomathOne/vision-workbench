# Vision Workbench

End-to-end Computer Vision MLOps Platform — from data cleaning to edge deployment.

## Quick Start

```bash
pip install -e ".[dev]"
vw --help
vw list tasks
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design document.

## Project Structure

```
vision-workbench/
├── src/vision_workbench/    # Main package
│   ├── core/                # Base abstractions (ABCs, Registry, Context, Config)
│   ├── pipeline/            # 8 pipeline stages (data → annotate → train → ... → deploy)
│   ├── detectors/           # Pre-trained detector wrappers (OpenCV, YOLO, MediaPipe, HF)
│   ├── models/              # Model registry, zoo, architectures, fusion modules
│   ├── data/                # Dataset catalog, schema, versioning
│   ├── tracking/            # Experiment tracking (MLflow/W&B integration)
│   ├── viz/                 # Visualization (annotations, curves, dashboards)
│   ├── serve/               # Inference microservice
│   └── cli/                 # Typer CLI
├── configs/                 # Example pipeline YAML configs
├── templates/               # Reusable pipeline templates/recipes
├── tests/                   # pytest test suite
├── notebooks/               # Jupyter notebooks
└── docs/                    # Documentation
```

## Commands

```bash
vw run <config.yaml>          # Execute a pipeline
vw list detectors             # List registered detectors
vw list tasks                 # List supported vision tasks
vw detect <image> -d yolo     # Quick inference
vw serve                      # Start web UI
```

## License

MIT
