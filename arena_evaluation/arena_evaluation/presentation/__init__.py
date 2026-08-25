"""Report rendering. The heavy members load on first access."""

__all__ = ["ReportBuilder", "VizManifest"]


def __getattr__(name: str) -> object:
    if name == "ReportBuilder":
        from .report_builder import ReportBuilder

        return ReportBuilder
    if name == "VizManifest":
        from .viz_manifest import VizManifest

        return VizManifest
    raise AttributeError(name)
