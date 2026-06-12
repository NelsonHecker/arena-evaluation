from __future__ import annotations

import pathlib
import base64
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from .base import BasePlotRenderer
from ...processing.map_registry import MapRegistry


class TrajectoryRenderer(BasePlotRenderer):
    PLOT_TYPE = "trajectory"

    def _load_map_image(self, map_name: str, run_dir: pathlib.Path | None = None):
        return MapRegistry.get_map(map_name, run_dir=run_dir)

    def _render_single_plot(self, pdf, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None = None) -> str:
        fig = go.Figure()
        diff_col = self.spec.differentiate or "planner"
        
        map_meta = None
        if self.spec.options.get("show_map", True) and map_name:
            map_meta = self._load_map_image(map_name, run_dir=run_dir)
            
        for _, row in pdf.iterrows():
            path = row["path"]
            if path is None or len(path) == 0:
                continue
                
            planner = row.get(diff_col, "unknown")
            episode = row.get("episode", 0)
            
            try:
                path_arr = np.array([list(p) for p in path])
            except Exception:
                continue
                
            if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                continue
                
            x = path_arr[:, 0]
            y = path_arr[:, 1]
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=f"{planner} (Ep {episode})",
                opacity=0.6
            ))
            
        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
            
        # Add map overlay if available
        layout_args = dict(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white",
            colorway=px.colors.qualitative.Pastel
        )
        
        if map_meta:
            # Read png and encode to base64
            png_path = map_meta["png_path"]
            try:
                with open(png_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                
                res = map_meta["resolution"]
                origin_x, origin_y = map_meta["origin"][:2]
                w_m = map_meta["width"] * res
                h_m = map_meta["height"] * res
                
                # Plotly expects coordinates for the corners
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{encoded}",
                        xref="x",
                        yref="y",
                        x=origin_x,
                        y=origin_y + h_m, # explicitly top-left of the image bounds
                        sizex=w_m,
                        sizey=h_m,
                        xanchor="left",
                        yanchor="top",
                        sizing="fill",
                        opacity=0.5,
                        layer="below"
                    )
                )
                
                # Optional: fix axis ranges to map bounds
                layout_args["xaxis"] = dict(range=[origin_x, origin_x + w_m])
                layout_args["yaxis"] = dict(range=[origin_y, origin_y + h_m], scaleanchor="x", scaleratio=1)
            except Exception as e:
                print(f"Failed to overlay map {map_name}: {e}")
                
        fig.update_layout(**layout_args)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _render_grid_plot(self, df_filtered: pl.DataFrame, valid_groups: list[str], run_dir: pathlib.Path | None) -> str | None:
        from plotly.subplots import make_subplots
        import numpy as np
        import base64
        
        # Group the data
        grouped_data = list(df_filtered.group_by(valid_groups))
        num_groups = len(grouped_data)
        if num_groups == 0:
            return None
            
        # Determine grid shape
        cols = 2
        rows = (num_groups + cols - 1) // cols
        
        # Create subplot titles
        subplot_titles = []
        for name, _ in grouped_data:
            if isinstance(name, tuple):
                title = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name))
            else:
                title = f"{valid_groups[0]}: {name}"
            subplot_titles.append(title)
            
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=False
        )
        
        diff_col = self.spec.differentiate or "planner"
        show_map = self.spec.options.get("show_map", True)
        
        seen_planners = set()
        
        for idx, (name, group_df) in enumerate(grouped_data, 1):
            r = (idx - 1) // cols + 1
            c = (idx - 1) % cols + 1
            
            pdf = group_df.to_pandas()
            
            # Map suffix for layout reference
            axis_suffix = f"{idx}" if idx > 1 else ""
            xref = f"x{axis_suffix}"
            yref = f"y{axis_suffix}"
            
            # Extract map_name
            map_name = None
            if "map" in valid_groups:
                map_idx = valid_groups.index("map")
                map_name = name[map_idx] if isinstance(name, tuple) else name
            elif "map" in pdf.columns:
                map_name = pdf["map"].iloc[0]
                
            map_meta = None
            if show_map and map_name:
                map_meta = self._load_map_image(map_name, run_dir=run_dir)
                
            # Add map image background
            if map_meta:
                try:
                    png_path = map_meta["png_path"]
                    with open(png_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                        
                    res = map_meta["resolution"]
                    origin_x, origin_y = map_meta["origin"][:2]
                    w_m = map_meta["width"] * res
                    h_m = map_meta["height"] * res
                    
                    fig.add_layout_image(
                        dict(
                            source=f"data:image/png;base64,{encoded}",
                            xref=xref,
                            yref=yref,
                            x=origin_x,
                            y=origin_y + h_m,
                            sizex=w_m,
                            sizey=h_m,
                            xanchor="left",
                            yanchor="top",
                            sizing="fill",
                            opacity=0.5,
                            layer="below"
                        )
                    )
                    
                    # Set axis ranges for this subplot
                    fig.update_layout({
                        f"xaxis{axis_suffix}": dict(range=[origin_x, origin_x + w_m]),
                        f"yaxis{axis_suffix}": dict(
                            range=[origin_y, origin_y + h_m],
                            scaleanchor=f"x{axis_suffix}",
                            scaleratio=1
                        )
                    })
                except Exception as e:
                    print(f"Failed to overlay map {map_name} on subplot {idx}: {e}")
                    
            for _, row in pdf.iterrows():
                path = row["path"]
                if path is None or len(path) == 0:
                    continue
                    
                planner = row.get(diff_col, "unknown")
                episode = row.get("episode", 0)
                
                try:
                    path_arr = np.array([list(p) for p in path])
                except Exception:
                    continue
                    
                if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                    continue
                    
                x = path_arr[:, 0]
                y = path_arr[:, 1]
                
                showlegend = planner not in seen_planners
                seen_planners.add(planner)
                
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name=planner,
                        legendgroup=planner,
                        showlegend=showlegend,
                        opacity=0.7
                    ),
                    row=r, col=c
                )
                
        height = 500 * rows
        fig.update_layout(
            title=self.spec.title,
            height=height,
            template="plotly_white",
            colorway=px.colors.qualitative.Pastel,
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _render_grid_seaborn(self, df_filtered: pl.DataFrame, valid_groups: list[str], out_path: pathlib.Path, run_dir: pathlib.Path | None) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        grouped_data = list(df_filtered.group_by(valid_groups))
        num_groups = len(grouped_data)
        if num_groups == 0:
            return
            
        cols = 2
        rows = (num_groups + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
        diff_col = self.spec.differentiate or "planner"
        
        for idx, (name, group_df) in enumerate(grouped_data):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            
            pdf = group_df.to_pandas()
            
            if isinstance(name, tuple):
                title = " | ".join(f"{col}: {n}" for col, n in zip(valid_groups, name))
            else:
                title = f"{valid_groups[0]}: {name}"
                
            ax.set_title(title)
            
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
                
                planner = row.get(diff_col, "unknown")
                ax.plot(x, y, label=planner, alpha=0.6)
                
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis('equal')
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="upper right")
                
        for idx in range(num_groups, rows * cols):
            r = idx // cols
            c = idx % cols
            fig.delaxes(axes[r, c])
            
        fig.suptitle(self.spec.title, fontsize=16)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        df_filtered = self._apply_filters(df)
        if "path" not in df_filtered.columns:
            return None
            
        run_dir = getattr(self, "run_dir", None)
            
        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]
                
            # Filter valid group columns
            valid_groups = [c for c in group_cols if c in df_filtered.columns]
            if not valid_groups:
                return self._render_single_plot(df_filtered.to_pandas(), "", None, run_dir=run_dir)
                
            return self._render_grid_plot(df_filtered, valid_groups, run_dir)
        else:
            pdf = df_filtered.to_pandas()
            map_name = pdf["map"].iloc[0] if "map" in pdf.columns and not pdf.empty else None
            return self._render_single_plot(pdf, "", map_name, run_dir=run_dir)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if "path" not in df_filtered.columns:
            return
            
        run_dir = getattr(self, "run_dir", None)
            
        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]
                
            valid_groups = [c for c in group_cols if c in df_filtered.columns]
            if not valid_groups:
                pdf = df_filtered.to_pandas()
                if not pdf.empty:
                    self.render_seaborn_single(pdf, out_path)
                return
                
            self._render_grid_seaborn(df_filtered, valid_groups, out_path, run_dir)
        else:
            pdf = df_filtered.to_pandas()
            if not pdf.empty:
                self.render_seaborn_single(pdf, out_path)

    def render_seaborn_single(self, pdf, out_path: pathlib.Path) -> None:
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
                
            planner = row.get(diff_col, "unknown")
            plt.plot(x, y, label=planner, alpha=0.6)
            
        plt.title(self.spec.title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
