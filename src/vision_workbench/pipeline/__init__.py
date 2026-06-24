"""Pipeline stages — auto-discovered for the orchestrator."""

import importlib

_STAGE_MODULES = [
    "vision_workbench.pipeline.data.stage",
    "vision_workbench.pipeline.annotate.stage",
    "vision_workbench.pipeline.train.stage",
    "vision_workbench.pipeline.validate.stage",
    "vision_workbench.pipeline.evaluate.stage",
    "vision_workbench.pipeline.optimize.stage",
    "vision_workbench.pipeline.export.stage",
    "vision_workbench.pipeline.deploy.stage",
]


def discover_stages() -> dict:
    """Import all stage modules and return name -> class mapping."""
    stages = {}
    for mod_name in _STAGE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and attr.__name__.endswith("Stage") and hasattr(attr, "name") and getattr(attr, "name"):
                    stages[attr.name] = attr
        except ImportError:
            pass
    return stages
