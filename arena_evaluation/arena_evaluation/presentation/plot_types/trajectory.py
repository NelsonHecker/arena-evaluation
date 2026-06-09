from __future__ import annotations

import pathlib
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from .base import BasePlotRenderer


class TrajectoryRenderer(BasePlotRenderer):
    PLOT_TYPE = "trajectory"

    def _load_map_image(self, map_name: str):
        return None

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if "path" not in df_filtered.columns:
            return None
            
        diff_col = self.spec.differentiate or "planner"
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        fig = go.Figure()
        
        for _, row in pdf.iterrows():
            path = row["path"]
            if path is None or len(path) == 0:
                continue
                
            planner = row.get(diff_col, "unknown")
            episode = row.get("episode", 0)
            
            # path is list of [x,y,yaw]
            try:
                path_arr = np.array([list(p) for p in path])
            except Exception:
                continue
                
            if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                continue
                
            x = path_arr[:, 0]
            y = path_arr[:, 1]
            
            if self.spec.options.get("center_at_origin", False):
                x = x - x[0]
                y = y - y[0]
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=f"{planner} (Ep {episode})",
                opacity=0.6
            ))
            
        fig.update_layout(
            title=self.spec.title,
            xaxis_title="X",
            yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1), # Make it square
            template="plotly_white",
            colorway=px.colors.qualitative.Pastel
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if "path" not in df_filtered.columns:
            return
            
        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 10))
        diff_col = self.spec.differentiate or "planner"
        
        for _, row in pdf.iterrows():
            path = row["path"]
            if path is None or len(path) == 0:
                continue
                
            try:
                path_arr = np.array([list(p) for p in path])
            except Exception:
                continue
                
            if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                continue
                
            x = path_arr[:, 0]
            y = path_arr[:, 1]
            
            if self.spec.options.get("center_at_origin", False):
                x = x - x[0]
                y = y - y[0]
                
            planner = row.get(diff_col, "unknown")
            
            plt.plot(x, y, label=planner, alpha=0.6)
            
        plt.title(self.spec.title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal') # Make it square
        
        # Deduplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
