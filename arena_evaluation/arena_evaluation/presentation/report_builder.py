from __future__ import annotations

import pathlib
from typing import Any
import polars as pl

from .viz_manifest import VizManifest
from .plotly_renderer import PlotlyRenderer
from .seaborn_renderer import SeabornRenderer
from ..processing.parquet_store import ParquetStore


class ReportBuilder:
    """
    Generates the final interactive HTML report and static PNG plots.
    """
    def __init__(self, benchmark_dir: pathlib.Path):
        self.benchmark_dir = benchmark_dir
        self.plots_dir = self.benchmark_dir / "plots"
        self.report_path = self.benchmark_dir / "report.html"
        self.manifest_path = self.benchmark_dir / "viz_manifest.yaml"
        
    def build(self) -> None:
        """Execute the report building process."""
        combined_metrics_path = self.benchmark_dir / "combined_metrics.parquet"
        metrics_path = self.benchmark_dir / "metrics.parquet"
        
        target_path = None
        if combined_metrics_path.exists():
            target_path = combined_metrics_path
        elif metrics_path.exists():
            target_path = metrics_path
            
        if target_path is None:
            print(f"Cannot generate report: neither combined_metrics.parquet nor metrics.parquet found in {self.benchmark_dir}.")
            return
            
        df, _ = ParquetStore.read(target_path)
        manifest = VizManifest.load(self.manifest_path)
        
        self.plots_dir.mkdir(exist_ok=True)
        
        plotly_renderer = PlotlyRenderer()
        seaborn_renderer = SeabornRenderer()
        
        html_plots = []
        
        for spec in manifest.plots:
            # Generate static PNG
            png_path = self.plots_dir / f"{spec.id}.png"
            try:
                seaborn_renderer.render(spec, df, png_path)
            except Exception as e:
                print(f"Warning: Failed to render static plot {spec.id}: {e}")
                
            # Generate interactive HTML chunk
            try:
                html_chunk = plotly_renderer.render(spec, df)
                if html_chunk:
                    html_plots.append((spec.layout_group, html_chunk))
            except Exception as e:
                print(f"Warning: Failed to render interactive plot {spec.id}: {e}")
                
        # Generate summary table
        summary_html = self._generate_summary_table(df)
                
        # Write final HTML
        html_content = self._assemble_html(summary_html, html_plots, self.benchmark_dir.name)
        with open(self.report_path, "w") as f:
            f.write(html_content)
            
        # Write local plotly.min.js to keep html file size small but work completely offline
        import plotly.offline
        js_path = self.benchmark_dir / "plotly.min.js"
        with open(js_path, "w") as f:
            f.write(plotly.offline.get_plotlyjs())
            
        print(f"Report generated successfully: {self.report_path}")
        print(f"Static plots saved to: {self.plots_dir}")

    def _generate_summary_table(self, df) -> str:
        if "planner" not in df.columns:
            return ""
            
        # Group by planner and calculate success rate, avg time, avg path length
        summary = df.group_by("planner").agg([
            pl.col("success").mean().alias("success_rate"),
            pl.col("time_to_goal").mean().alias("avg_time"),
            pl.col("path_length").mean().alias("avg_path_length"),
            pl.col("collision_amount").mean().alias("avg_collisions")
        ]).to_pandas()
        
        # Format columns
        if "success_rate" in summary.columns:
            summary["success_rate"] = (summary["success_rate"] * 100).map("{:.1f}%".format)
        if "avg_time" in summary.columns:
            summary["avg_time"] = summary["avg_time"].map("{:.2f}s".format)
        if "avg_path_length" in summary.columns:
            summary["avg_path_length"] = summary["avg_path_length"].map("{:.2f}m".format)
        if "avg_collisions" in summary.columns:
            summary["avg_collisions"] = summary["avg_collisions"].map("{:.2f}".format)
            
        return summary.to_html(index=False, classes="table table-striped table-hover")

    def _assemble_html(self, summary_html: str, plot_htmls: list[tuple[str|None, str]], benchmark_id: str) -> str:
        """Template for the final HTML report."""
        
        # Group plots by layout_group
        grouped_plots = {}
        ordered_groups = []
        for group, html in plot_htmls:
            # If no group, assign a unique single-item group
            key = group if group else f"__ungrouped_{len(ordered_groups)}"
            if key not in grouped_plots:
                grouped_plots[key] = []
                ordered_groups.append(key)
            grouped_plots[key].append(html)
            
        final_plot_htmls = []
        for key in ordered_groups:
            plots = grouped_plots[key]
            if key.startswith("__ungrouped_") or len(plots) == 1:
                final_plot_htmls.append(f'<div class="plot-container">{plots[0]}</div>')
            else:
                grid_items = "".join([f'<div class="plot-item">{p}</div>' for p in plots])
                final_plot_htmls.append(f'<div class="plot-group"><h2>{key.replace("_", " ").title()}</h2><div class="plot-grid">{grid_items}</div></div>')
                
        plots_joined = "\n<hr>\n".join(final_plot_htmls)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Arena Evaluation Report - {benchmark_id}</title>
            <script src="plotly.min.js"></script>
            <style>
                * {{ box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary-table {{ margin-bottom: 40px; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                
                .plot-container {{ height: 600px; margin-bottom: 50px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }}
                .plot-container > div {{ height: 100%; width: 100%; }}
                
                .plot-group {{ margin-bottom: 50px; background: #f8f9fa; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .plot-item {{ height: 450px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 10px; }}
                .plot-item > div {{ height: 100%; width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Arena Evaluation Report</h1>
            <p><strong>Benchmark ID:</strong> {benchmark_id}</p>
            
            <h2>Summary</h2>
            <div class="summary-table">
                {summary_html}
            </div>
            
            <h2>Metrics Analysis</h2>
            <div class="plots-section">
                {plots_joined}
            </div>
            
            <div style="margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em;">
                <p>Generated by Arena Evaluation Pipeline</p>
            </div>
        </body>
        </html>
        """
        return html
