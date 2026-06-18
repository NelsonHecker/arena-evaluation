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
            
        # Extract planners available in this subplot
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        seen_planners = set()
        
        for planner in planners:
            if diff_col in pdf.columns:
                planner_df = pdf[pdf[diff_col] == planner]
            else:
                planner_df = pdf
                
            # Sort by episode chronologically
            if "episode" in planner_df.columns:
                planner_df = planner_df.sort_values("episode")
                
            all_x = []
            all_y = []
            
            for _, row in planner_df.iterrows():
                path = row["path"]
                if path is None or len(path) == 0:
                    continue
                    
                try:
                    path_arr = np.array([list(p) for p in path])
                except Exception:
                    continue
                    
                if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                    continue
                    
                all_x.extend(path_arr[:, 0])
                all_y.extend(path_arr[:, 1])
                
            if not all_x:
                continue
                
            x_arr = np.array(all_x)
            y_arr = np.array(all_y)
            
            # Split the continuous path by teleport jumps (> 0.5m)
            dists = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
            jumps = np.where(dists > 0.5)[0]
            split_indices = jumps + 1
            
            segments_x = np.split(x_arr, split_indices)
            segments_y = np.split(y_arr, split_indices)
            
            final_x = []
            final_y = []
            
            for seg_x, seg_y in zip(segments_x, segments_y):
                if len(seg_x) < 2:
                    continue
                    
                seg_len = np.sum(np.sqrt(np.diff(seg_x)**2 + np.diff(seg_y)**2))
                if seg_len >= 0.2:
                    final_x.extend(seg_x)
                    final_x.append(np.nan)  # Disconnect segments to prevent lines warping across map
                    final_y.extend(seg_y)
                    final_y.append(np.nan)
                    
            if not final_x:
                continue
                
            showlegend = planner not in seen_planners
            seen_planners.add(planner)
            
            fig.add_trace(
                go.Scatter(
                    x=final_x, y=final_y,
                    mode='lines',
                    name=planner,
                    legendgroup=planner,
                    showlegend=showlegend,
                    opacity=0.7
                )
            )
            
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
            colorway=px.colors.qualitative.Pastel,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
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
        return self._inject_slider_js(fig)

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
            
            planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
            
            for planner in planners:
                if diff_col in pdf.columns:
                    planner_df = pdf[pdf[diff_col] == planner]
                else:
                    planner_df = pdf
                    
                if "episode" in planner_df.columns:
                    planner_df = planner_df.sort_values("episode")
                    
                all_x = []
                all_y = []
                
                for _, row in planner_df.iterrows():
                    path = row["path"]
                    if path is None or len(path) == 0:
                        continue
                        
                    try:
                        path_arr = np.array([list(p) for p in path])
                    except Exception:
                        continue
                        
                    if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                        continue
                        
                    all_x.extend(path_arr[:, 0])
                    all_y.extend(path_arr[:, 1])
                    
                if not all_x:
                    continue
                    
                x_arr = np.array(all_x)
                y_arr = np.array(all_y)
                
                dists = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
                jumps = np.where(dists > 0.5)[0]
                split_indices = jumps + 1
                
                segments_x = np.split(x_arr, split_indices)
                segments_y = np.split(y_arr, split_indices)
                
                final_x = []
                final_y = []
                
                for seg_x, seg_y in zip(segments_x, segments_y):
                    if len(seg_x) < 2:
                        continue
                        
                    seg_len = np.sum(np.sqrt(np.diff(seg_x)**2 + np.diff(seg_y)**2))
                    if seg_len >= 0.2:
                        final_x.extend(seg_x)
                        final_x.append(np.nan)
                        final_y.extend(seg_y)
                        final_y.append(np.nan)
                        
                if not final_x:
                    continue
                    
                ax.plot(final_x, final_y, label=planner, alpha=0.6)
                
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

    def _inject_slider_js(self, fig: go.Figure) -> str:
        import uuid
        plot_id = f"traj_plot_{uuid.uuid4().hex}"
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=plot_id, config={'responsive': True})
        
        custom_js = f"""
        <div style="margin-top: 15px; padding: 10px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <label for="slider_{plot_id}" style="font-weight: 600; color: #1e293b;">Time Progression (s)</label>
                <span style="font-family: monospace; font-weight: bold; background: #e2e8f0; padding: 2px 8px; border-radius: 4px;" id="val_{plot_id}">Max</span>
            </div>
            <input type="range" id="slider_{plot_id}" min="0" max="100" value="100" style="width: 100%; cursor: pointer;">
        </div>
        <script>
            setTimeout(function() {{
                var plotDiv = document.getElementById("{plot_id}");
                if (!plotDiv || !plotDiv.data) return;
                
                plotDiv._originalData = plotDiv.data.map(t => ({{
                    x: t.x ? Array.from(t.x) : [], 
                    y: t.y ? Array.from(t.y) : []
                }}));
                
                var maxLen = 0;
                plotDiv._originalData.forEach(t => {{
                    if (t.x.length > maxLen) maxLen = t.x.length;
                }});
                
                var slider = document.getElementById("slider_{plot_id}");
                var label = document.getElementById("val_{plot_id}");
                
                function formatTime(steps) {{
                    var totalMs = steps * 100; // assuming dt = 0.1s
                    var h = Math.floor(totalMs / 3600000);
                    var m = Math.floor((totalMs % 3600000) / 60000);
                    var s = Math.floor((totalMs % 60000) / 1000);
                    var ms = totalMs % 1000;
                    return String(h).padStart(2, '0') + ':' + 
                           String(m).padStart(2, '0') + ':' + 
                           String(s).padStart(2, '0') + ':' + 
                           String(ms).padStart(3, '0');
                }}
                
                if (maxLen > 0) {{
                    slider.max = maxLen;
                    slider.value = maxLen;
                    var maxStr = formatTime(maxLen);
                    label.innerText = maxStr + " / " + maxStr;
                    
                    slider.addEventListener("input", function(e) {{
                        var limit = parseInt(e.target.value);
                        label.innerText = formatTime(limit) + " / " + maxStr;
                        
                        // Mutate directly and redraw for 100% reliability
                        plotDiv.data.forEach((trace, i) => {{
                            trace.x = plotDiv._originalData[i].x.slice(0, limit);
                            trace.y = plotDiv._originalData[i].y.slice(0, limit);
                        }});
                        
                        Plotly.redraw(plotDiv);
                    }});
                }} else {{
                    slider.disabled = true;
                }}
            }}, 500);
        </script>
        """
        return fig_html + custom_js

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
                
            htmls = []
            for name, group_df in df_filtered.group_by(valid_groups):
                pdf = group_df.to_pandas()
                # Create a title suffix from the group names
                if isinstance(name, tuple):
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name))
                else:
                    suffix = f"{valid_groups[0]}: {name}"
                
                # Extract map_name
                map_name = None
                if "map" in valid_groups:
                    map_idx = valid_groups.index("map")
                    map_name = name[map_idx] if isinstance(name, tuple) else name
                elif "map" in pdf.columns:
                    map_name = pdf["map"].iloc[0] if not pdf.empty else None
                    
                html = self._render_single_plot(pdf, suffix, map_name, run_dir=run_dir)
                if html:
                    htmls.append(html)
            return htmls
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
