"""Pipeline stages — flat 5-stage closed-loop workflow."""

import importlib

_STAGE_MODULES = [
    "vision_workbench.pipeline.data_stage",
    "vision_workbench.pipeline.train_stage",
    "vision_workbench.pipeline.validate_stage",
    "vision_workbench.pipeline.export_stage",
    "vision_workbench.pipeline.deploy_stage",
]

_STAGE_CLASS_NAMES = ["DataStage", "TrainStage", "ValidateStage", "ExportStage", "DeployStage"]


def discover_stages() -> dict:
    """Import all stage modules and return name -> class mapping."""
    stages = {}
    for mod_name, cls_name in zip(_STAGE_MODULES, _STAGE_CLASS_NAMES):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls and getattr(cls, "name", None):
                stages[cls.name] = cls
        except ImportError:
            pass
    return stages
