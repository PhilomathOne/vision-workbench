"""Gradio-based web UI for Vision Workbench."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vision_workbench.core.registry import detector_registry
from vision_workbench.viz.annotate import annotate


def create_ui():
    """Build and return the Gradio Blocks UI."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for the web UI. "
            "Install with: pip install vision-workbench[serve]"
        )

    # Discover available detectors
    detectors = list(detector_registry.list().keys())
    if not detectors:
        detectors = ["opencv_haar_face"]

    with gr.Blocks(title="Vision Workbench", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            """
            # 🔍 Vision Workbench
            ### End-to-end Computer Vision MLOps Platform
            """
        )

        with gr.Tab("Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image", type="numpy")
                    detector_dd = gr.Dropdown(
                        choices=detectors,
                        value=detectors[0] if detectors else None,
                        label="Detector",
                    )
                    detect_btn = gr.Button("Detect", variant="primary")
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Detection Results", type="numpy")
                    result_json = gr.JSON(label="Raw Results")

            detect_btn.click(
                fn=run_detection,
                inputs=[input_image, detector_dd],
                outputs=[output_image, result_json],
            )

        with gr.Tab("Pipeline"):
            gr.Markdown(
                """
                ### Pipeline Configuration
                Upload a YAML pipeline config or use the CLI:
                ```bash
                vw run configs/full_pipeline.yaml
                ```
                """
            )
            config_file = gr.File(label="Pipeline Config (YAML)")
            run_btn = gr.Button("Run Pipeline")
            pipeline_output = gr.Textbox(label="Output", lines=10)
            run_btn.click(fn=run_pipeline, inputs=[config_file], outputs=[pipeline_output])

        with gr.Tab("Model Zoo"):
            gr.Markdown("### Pre-trained Model Zoo")
            from vision_workbench.models.zoo import KNOWN_MODELS, ModelZoo

            zoo = ModelZoo()
            cached = zoo.list()

            model_data = []
            for name, info in KNOWN_MODELS.items():
                model_data.append([
                    name,
                    info.get("task", "—"),
                    info.get("framework", "—"),
                    f"{info.get('size_mb', '?')} MB",
                    "✅" if name in cached else "⬇️",
                ])

            gr.Dataframe(
                headers=["Model", "Task", "Framework", "Size", "Cached"],
                value=model_data,
                label="Available Models",
            )

    return ui


def run_detection(
    image: Optional[np.ndarray],
    detector_name: str,
) -> tuple[Optional[np.ndarray], dict]:
    """Run detection on the uploaded image."""
    if image is None:
        return None, {"error": "No image provided"}

    try:
        detector_cls = detector_registry.get(detector_name)
        detector = detector_cls()
        detector.initialize()
        result = detector.process(image)
        annotated = annotate(image, result)
        detector.cleanup()

        result_dict = {
            "detector": detector_name,
            "boxes": len(result.boxes),
            "processing_time_ms": round(result.processing_time_ms, 1),
            "detections": [
                {"class": b.class_name, "confidence": round(b.confidence, 3) if b.confidence else None, "bbox": [b.x1, b.y1, b.x2, b.y2]}
                for b in result.boxes[:20]
            ],
        }
        return annotated, result_dict
    except Exception as e:
        return None, {"error": str(e)}


def run_pipeline(config_file) -> str:
    """Run a pipeline from an uploaded config."""
    if config_file is None:
        return "No config file provided. Please upload a YAML config."
    try:
        from vision_workbench.core.config import PipelineConfig

        config = PipelineConfig.from_yaml(config_file.name)
        return f"Config loaded: {config.name}\nStages: {config.stages}\n\nRun via CLI:\nvw run {config_file.name}"
    except Exception as e:
        return f"Error: {e}"


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio web UI."""
    ui = create_ui()
    ui.launch(server_name=host, server_port=port, share=share)
