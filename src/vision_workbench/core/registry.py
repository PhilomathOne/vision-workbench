"""Generic plugin registry for component discovery.

Provides a unified registration pattern used by detectors, format
converters, training frameworks, exporters, and architecture components.
"""

from typing import Callable, ClassVar, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Thread-safe generic registry for named component classes.

    Usage::

        detector_registry = Registry[BaseDetector]("detector")

        @detector_registry.register("yolov8")
        class YOLOv8Detector(BaseDetector):
            ...

        cls = detector_registry.get("yolov8")
        all_detectors = detector_registry.list()
    """

    def __init__(self, kind: str) -> None:
        self.kind: str = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str, **metadata: object) -> Callable[[type[T]], type[T]]:
        """Decorator to register a component class.

        Args:
            name: Unique name for this component.
            **metadata: Arbitrary key-value tags stored on the class
                       for filtering (e.g., task="detection", framework="yolo").
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = cls
            # Attach metadata to the class for registry queries
            if not hasattr(cls, "_registry_metadata"):
                cls._registry_metadata = {}  # type: ignore[attr-defined]
            cls._registry_metadata.update(metadata)  # type: ignore[attr-defined]
            return cls

        return decorator

    def register_direct(self, name: str, cls: type[T], **metadata: object) -> None:
        """Programmatic registration (no decorator needed)."""
        if name in self._entries:
            raise KeyError(f"{self.kind} '{name}' is already registered")
        self._entries[name] = cls
        if not hasattr(cls, "_registry_metadata"):
            cls._registry_metadata = {}  # type: ignore[attr-defined]
        cls._registry_metadata.update(metadata)  # type: ignore[attr-defined]

    def get(self, name: str) -> type[T]:
        """Look up a component by name.

        Raises:
            RegistryError: If the name is not found.
        """
        from vision_workbench.core.exceptions import RegistryError

        if name not in self._entries:
            available = list(self._entries.keys())
            raise RegistryError(
                f"Unknown {self.kind} '{name}'. Available: {available}"
            )
        return self._entries[name]

    def list(self) -> dict[str, type[T]]:
        """Return all registered entries."""
        return dict(self._entries)

    def list_by(self, **filters: object) -> dict[str, type[T]]:
        """Return entries filtered by metadata tags.

        Example::

            registry.list_by(task="object_detection")
        """
        result: dict[str, type[T]] = {}
        for name, cls in self._entries.items():
            meta: dict = getattr(cls, "_registry_metadata", {})
            if all(meta.get(str(k)) == v for k, v in filters.items()):
                result[name] = cls
        return result

    def discover_entry_points(self, group: str) -> None:
        """Load plugins from pyproject.toml entry_points.

        This allows third-party packages to register components by
        declaring an entry point under the given group name.
        """
        try:
            import importlib.metadata as md
        except ImportError:
            return

        for ep in md.entry_points(group=group):
            try:
                ep.load()  # triggers @register decorator
            except Exception:
                # Skip broken plugins — don't crash the whole registry
                pass

    def clear(self) -> None:
        """Remove all entries (primarily for testing)."""
        self._entries.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(kind='{self.kind}', entries={len(self._entries)})"


# Global registry instances — populated as modules are imported
detector_registry: Registry = Registry("detector")
format_registry: Registry = Registry("format_converter")
framework_registry: Registry = Registry("framework")
exporter_registry: Registry = Registry("exporter")
backbone_registry: Registry = Registry("backbone")
neck_registry: Registry = Registry("neck")
head_registry: Registry = Registry("head")
attention_registry: Registry = Registry("attention")
fusion_registry: Registry = Registry("fusion")
architecture_registry: Registry = Registry("architecture")
