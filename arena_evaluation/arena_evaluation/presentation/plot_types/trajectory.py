from __future__ import annotations

import pathlib
import base64
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from .base import BasePlotRenderer
from ..color_utils import get_color_palette
from ...processing.map_registry import MapRegistry


class TrajectoryRenderer(BasePlotRenderer):
    PLOT_TYPE = "trajectory"
    generate_gifs: bool = False

    def _load_map_image(self, map_name: str, run_dir: pathlib.Path | None = None):
        return MapRegistry.get_map(map_name, run_dir=run_dir)

    def _render_single_plot(self, pdf, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None = None) -> str:
        fig = go.Figure()
        diff_col = self.spec.differentiate or "planner"
        
        map_meta = None
        if self.spec.options.get("show_map", True) and map_name:
            map_meta = self._load_map_image(map_name, run_dir=run_dir)
            
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        seen_planners = set()
        
        for planner in planners:
            if diff_col in pdf.columns:
                planner_df = pdf[pdf[diff_col] == planner]
            else:
                planner_df = pdf
                
            if "episode" in planner_df.columns:
                planner_df = planner_df.sort_values("episode")
                
            overlay_markers = self.spec.options.get("overlay_markers", True)
            
            all_x_by_agent = []
            all_y_by_agent = []
            
            if overlay_markers:
                starts_x_by_agent = []
                starts_y_by_agent = []
                starts_idx_by_agent = []
                goals_x_by_agent = []
                goals_y_by_agent = []
                goals_idx_by_agent = []
                col_x_by_agent = []
                col_y_by_agent = []
                col_idx_by_agent = []
            
            data_key = self.spec.data_key or "path"
            for _, row in planner_df.iterrows():
                path = row.get(data_key)
                if path is None or len(path) == 0:
                    continue
                    
                is_collision = row.get("result") == "COLLISION"
                    
                paths_to_process = []
                try:
                    first_elem = path[0]
                    if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)):
                        for sub_path in path:
                            if sub_path is not None and len(sub_path) > 0:
                                paths_to_process.append(np.array([list(p) for p in sub_path]))
                    else:
                        paths_to_process.append(np.array([list(p) for p in path]))
                except Exception:
                    continue
                    
                for k, path_arr in enumerate(paths_to_process):
                    if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                        continue
                        
                    while len(all_x_by_agent) <= k:
                        all_x_by_agent.append([])
                        all_y_by_agent.append([])
                        if overlay_markers:
                            starts_x_by_agent.append([])
                            starts_y_by_agent.append([])
                            starts_idx_by_agent.append([])
                            goals_x_by_agent.append([])
                            goals_y_by_agent.append([])
                            goals_idx_by_agent.append([])
                            col_x_by_agent.append([])
                            col_y_by_agent.append([])
                            col_idx_by_agent.append([])
                            
                    if overlay_markers:
                        current_len = len(all_x_by_agent[k])
                        valid_mask = ~np.isnan(path_arr[:, 0])
                        if np.any(valid_mask):
                            clean_indices = np.where(valid_mask)[0]
                            
                            dists = np.sqrt(np.diff(path_arr[clean_indices, 0])**2 + np.diff(path_arr[clean_indices, 1])**2)
                            jumps = np.where(dists > 3.0)[0]
                            segment_starts = np.concatenate([[0], jumps + 1])
                            segment_ends = np.concatenate([jumps, [len(dists)]])
                            
                            first_clean_idx = segment_starts[-1]
                            for s_idx, e_idx in zip(segment_starts, segment_ends):
                                if e_idx > s_idx:
                                    segment_length = np.sum(dists[s_idx:e_idx])
                                    if segment_length > 0.1:
                                        first_clean_idx = s_idx
                                        break
                                
                            first_idx = int(clean_indices[first_clean_idx])
                            last_idx = int(clean_indices[-1])
                            
                            starts_x_by_agent[k].append(path_arr[first_idx, 0])
                            starts_y_by_agent[k].append(path_arr[first_idx, 1])
                            starts_idx_by_agent[k].append(current_len + first_idx)
                            
                            if is_collision:
                                col_x_by_agent[k].append(path_arr[last_idx, 0])
                                col_y_by_agent[k].append(path_arr[last_idx, 1])
                                col_idx_by_agent[k].append(current_len + last_idx)
                            else:
                                goals_x_by_agent[k].append(path_arr[last_idx, 0])
                                goals_y_by_agent[k].append(path_arr[last_idx, 1])
                                goals_idx_by_agent[k].append(current_len + last_idx)
                                
                    all_x_by_agent[k].extend(path_arr[:, 0])
                    all_y_by_agent[k].extend(path_arr[:, 1])
                    
            palette = get_color_palette()
            planner_idx = list(planners).index(planner) if planner in planners else 0
            planner_color = palette[planner_idx % len(palette)]

            for k in range(len(all_x_by_agent)):
                if not all_x_by_agent[k]:
                    continue
                    
                x_arr = np.array(all_x_by_agent[k], dtype=float)
                y_arr = np.array(all_y_by_agent[k], dtype=float)
                
                dists = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
                jumps = np.where((dists > 3.0) & ~np.isnan(dists))[0]
                split_indices = jumps + 1
                
                final_x = x_arr.copy()
                final_y = y_arr.copy()
                if len(split_indices) > 0:
                    final_x[split_indices] = np.nan
                    final_y[split_indices] = np.nan
                    
                final_x = final_x.tolist()
                final_y = final_y.tolist()
                
                if len(all_x_by_agent) > 1:
                    trace_name = f"{planner} - Agent {k}"
                    legendgroup = planner
                    showlegend = planner not in seen_planners
                    seen_planners.add(planner)
                    opacity = 0.5
                else:
                    trace_name = planner
                    legendgroup = planner
                    showlegend = planner not in seen_planners
                    seen_planners.add(planner)
                    opacity = 0.8
                
                fig.add_trace(
                    go.Scatter(
                        x=final_x, y=final_y,
                        mode='lines',
                        name=trace_name,
                        legendgroup=legendgroup,
                        showlegend=showlegend,
                        opacity=opacity,
                        line=dict(color=planner_color),
                        hovertemplate=f"<b>{{planner}}</b><br>{{trace_name}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>"
                    )
                )
                
                if overlay_markers:
                    if starts_x_by_agent[k]:
                        fig.add_trace(go.Scatter(
                            x=starts_x_by_agent[k], y=starts_y_by_agent[k],
                            mode='markers',
                            marker=dict(symbol='circle', size=8, color='#00bfb2'),
                            name=f"{planner} Starts",
                            legendgroup=planner,
                            showlegend=False,
                            opacity=0.8,
                            hovertemplate=f"<b>{{planner}}</b><br>Start<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                            customdata=starts_idx_by_agent[k]
                        ))
                    if goals_x_by_agent[k]:
                        fig.add_trace(go.Scatter(
                            x=goals_x_by_agent[k], y=goals_y_by_agent[k],
                            mode='markers',
                            marker=dict(symbol='star', size=12, color='#ffc845'),
                            name=f"{planner} Goals",
                            legendgroup=planner,
                            showlegend=False,
                            opacity=0.9,
                            hovertemplate=f"<b>{{planner}}</b><br>Goal<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                            customdata=goals_idx_by_agent[k]
                        ))
                    if col_x_by_agent[k]:
                        fig.add_trace(go.Scatter(
                            x=col_x_by_agent[k], y=col_y_by_agent[k],
                            mode='markers',
                            marker=dict(symbol='x', size=10, color='#d3273e', line=dict(width=2)),
                            name=f"{planner} Collisions",
                            legendgroup=planner,
                            showlegend=False,
                            opacity=1.0,
                            hovertemplate=f"<b>{{planner}}</b><br>Collision<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                            customdata=col_idx_by_agent[k]
                        ))
            
        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
            
        layout_args = dict(
            title=title,
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        
        if map_meta:
            png_path = map_meta["png_path"]
            try:
                with open(png_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                
                res = map_meta["resolution"]
                w_m = map_meta["width"] * res
                h_m = map_meta["height"] * res
                
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{encoded}",
                        xref="x",
                        yref="y",
                        x=0,
                        y=h_m,
                        sizex=w_m,
                        sizey=h_m,
                        xanchor="left",
                        yanchor="top",
                        sizing="fill",
                        opacity=0.5,
                        layer="below"
                    )
                )
                
                layout_args["xaxis"] = dict(range=[0, w_m])
                layout_args["yaxis"] = dict(range=[0, h_m], scaleanchor="x", scaleratio=1)
            except Exception as e:
                print(f"Failed to overlay map {map_name}: {e}")
                
        fig.update_layout(**layout_args)
        return self._inject_slider_js(fig)

    def _generate_gif(self, fig, lines_data, markers_info, out_path: pathlib.Path):
        import matplotlib.animation as animation
        
        max_len = max([len(data[1]) for data in lines_data]) if lines_data else 0
        if max_len == 0:
            return
            
        step = max(1, max_len // 100)
        frames = list(range(0, max_len, step))
        if frames[-1] != max_len:
            frames.append(max_len)
            
        def update(frame):
            artists = []
            for line, x, y in lines_data:
                line.set_data(x[:frame], y[:frame])
                artists.append(line)
                
            if markers_info:
                data = markers_info["data"]
                
                s_pts = [[m["x"], m["y"]] for m in data if m["type"] == "start" and frame >= m["frame"]]
                if markers_info.get("starts"):
                    markers_info["starts"].set_offsets(s_pts if s_pts else np.empty((0, 2)))
                    artists.append(markers_info["starts"])
                    
                g_pts = [[m["x"], m["y"]] for m in data if m["type"] == "goal" and frame >= m["frame"]]
                if markers_info.get("goals"):
                    markers_info["goals"].set_offsets(g_pts if g_pts else np.empty((0, 2)))
                    artists.append(markers_info["goals"])
                    
                c_pts = [[m["x"], m["y"]] for m in data if m["type"] == "collision" and frame >= m["frame"]]
                if markers_info.get("cols"):
                    markers_info["cols"].set_offsets(c_pts if c_pts else np.empty((0, 2)))
                    artists.append(markers_info["cols"])
                    
            return artists
            
        for line, _, _ in lines_data:
            line.set_data([], [])
            
        if markers_info:
            if markers_info.get("starts"): markers_info["starts"].set_offsets(np.empty((0, 2)))
            if markers_info.get("goals"): markers_info["goals"].set_offsets(np.empty((0, 2)))
            if markers_info.get("cols"): markers_info["cols"].set_offsets(np.empty((0, 2)))
            
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        try:
            ani.save(out_path, writer='pillow', fps=20)
        except Exception as e:
            print(f"Failed to save GIF {out_path}: {e}")

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
                plotDiv.data.forEach(t => {{
                    if (t.customdata && t.customdata.length > 0) {{
                        let cdata = Array.isArray(t.customdata[0]) ? t.customdata.map(d => d[0]) : t.customdata;
                        let m = Math.max(...cdata);
                        if (m > maxLen) maxLen = m;
                    }} else if (t.x && t.x.length > maxLen) {{
                        maxLen = t.x.length;
                    }}
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
                            if (trace.customdata && trace.customdata.length > 0) {{
                                let new_x = [];
                                let new_y = [];
                                let cdata = Array.isArray(trace.customdata[0]) ? trace.customdata.map(d => d[0]) : trace.customdata;
                                for (let j = 0; j < cdata.length; j++) {{
                                    if (limit >= cdata[j]) {{
                                        new_x.push(plotDiv._originalData[i].x[j]);
                                        new_y.push(plotDiv._originalData[i].y[j]);
                                    }}
                                }}
                                trace.x = new_x;
                                trace.y = new_y;
                            }} else {{
                                trace.x = plotDiv._originalData[i].x.slice(0, limit);
                                trace.y = plotDiv._originalData[i].y.slice(0, limit);
                            }}
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
        data_key = self.spec.data_key or "path"
        if data_key not in df_filtered.columns:
            return None
            
        run_dir = self.run_dir
            
        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]
                
            valid_groups = [c for c in group_cols if c in df_filtered.columns]
            if not valid_groups:
                return self._render_single_plot(df_filtered.to_pandas(), "", None, run_dir=run_dir)
                
            htmls = []
            for name, group_df in df_filtered.group_by(valid_groups):
                pdf = group_df.to_pandas()
                if isinstance(name, tuple):
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name))
                else:
                    suffix = f"{valid_groups[0]}: {name}"
                
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
        data_key = self.spec.data_key or "path"
        if data_key not in df_filtered.columns:
            return
            
        run_dir = self.run_dir
            
        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]
                
            valid_groups = [c for c in group_cols if c in df_filtered.columns]
            if not valid_groups:
                pdf = df_filtered.to_pandas()
                if not pdf.empty:
                    self.render_seaborn_single(pdf, out_path, "", None, run_dir)
                return
                
            for name, group_df in df_filtered.group_by(valid_groups):
                pdf = group_df.to_pandas()
                if isinstance(name, tuple):
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name))
                    file_suffix = "_".join(str(n).replace(" ", "_").replace("/", "_") for n in name)
                else:
                    suffix = f"{valid_groups[0]}: {name}"
                    file_suffix = str(name).replace(" ", "_").replace("/", "_")
                    
                map_name = None
                if "map" in valid_groups:
                    map_idx = valid_groups.index("map")
                    map_name = name[map_idx] if isinstance(name, tuple) else name
                elif "map" in pdf.columns:
                    map_name = pdf["map"].iloc[0] if not pdf.empty else None
                    
                group_out_path = out_path.with_name(f"{out_path.stem}_{file_suffix}{out_path.suffix}")
                self.render_seaborn_single(pdf, group_out_path, suffix, map_name, run_dir)
        else:
            pdf = df_filtered.to_pandas()
            if not pdf.empty:
                map_name = pdf["map"].iloc[0] if "map" in pdf.columns and not pdf.empty else None
                self.render_seaborn_single(pdf, out_path, "", map_name, run_dir)

    def render_seaborn_single(self, pdf, out_path: pathlib.Path, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        plt.figure(figsize=(10, 10))
        
        map_meta = None
        if self.spec.options.get("show_map", True) and map_name:
            map_meta = self._load_map_image(map_name, run_dir=run_dir)
            
        if map_meta:
            try:
                img = mpimg.imread(map_meta["png_path"])
                res = map_meta["resolution"]
                w_m = map_meta["width"] * res
                h_m = map_meta["height"] * res
                plt.imshow(img, extent=[0, w_m, 0, h_m], alpha=0.5, origin='upper', zorder=0)
                plt.xlim(0, w_m)
                plt.ylim(0, h_m)
            except Exception as e:
                print(f"Failed to overlay map {map_name} in seaborn: {e}")
                
        diff_col = self.spec.differentiate or "planner"
        
        overlay_markers = self.spec.options.get("overlay_markers", True)
        
        lines_data = []
        markers_data = []
        
        all_x_by_agent = {}
        starts_x, starts_y = [], []
        goals_x, goals_y = [], []
        col_x, col_y = [], []
        
        data_key = self.spec.data_key or "path"
        
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        seen_planners = set()
        
        for planner in planners:
            if diff_col in pdf.columns:
                planner_df = pdf[pdf[diff_col] == planner]
            else:
                planner_df = pdf
                
            if "episode" in planner_df.columns:
                planner_df = planner_df.sort_values("episode")
                
            if planner not in all_x_by_agent:
                all_x_by_agent[planner] = []
                
            for _, row in planner_df.iterrows():
                path = row.get(data_key)
                if path is None or len(path) == 0:
                    continue
                    
                is_collision = row.get("result") == "COLLISION"
                    
                paths_to_process = []
                try:
                    first_elem = path[0]
                    if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)):
                        for sub_path in path:
                            if sub_path is not None and len(sub_path) > 0:
                                paths_to_process.append(np.array([list(p) for p in sub_path]))
                    else:
                        paths_to_process.append(np.array([list(p) for p in path]))
                except Exception:
                    continue
                    
                for k, path_arr in enumerate(paths_to_process):
                    if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                        continue
                        
                    while len(all_x_by_agent[planner]) <= k:
                        all_x_by_agent[planner].append({"x": [], "y": []})
                        
                    current_len = len(all_x_by_agent[planner][k]["x"])
                    
                    if overlay_markers:
                        valid_mask = ~np.isnan(path_arr[:, 0])
                        if np.any(valid_mask):
                            clean_indices = np.where(valid_mask)[0]
                            
                            # Find actual start avoiding initial spawn jumps
                            dists = np.sqrt(np.diff(path_arr[clean_indices, 0])**2 + np.diff(path_arr[clean_indices, 1])**2)
                            jumps = np.where(dists > 3.0)[0]
                            segment_starts = np.concatenate([[0], jumps + 1])
                            segment_ends = np.concatenate([jumps, [len(dists)]])
                            
                            first_clean_idx = segment_starts[-1]
                            for s_idx, e_idx in zip(segment_starts, segment_ends):
                                if e_idx > s_idx:
                                    segment_length = np.sum(dists[s_idx:e_idx])
                                    if segment_length > 0.1:
                                        first_clean_idx = s_idx
                                        break
                                
                            first_idx = int(clean_indices[first_clean_idx])
                            last_idx = int(clean_indices[-1])
                            
                            # True start/goal from dataframe
                            true_start = row.get("start")
                            true_goal = row.get("goal")
                            
                            if isinstance(true_start, (list, tuple, np.ndarray)) and len(true_start) >= 2:
                                starts_x.append(true_start[0])
                                starts_y.append(true_start[1])
                                markers_data.append({"frame": current_len + first_idx, "x": true_start[0], "y": true_start[1], "type": "start"})
                            else:
                                starts_x.append(path_arr[first_idx, 0])
                                starts_y.append(path_arr[first_idx, 1])
                                markers_data.append({"frame": current_len + first_idx, "x": path_arr[first_idx, 0], "y": path_arr[first_idx, 1], "type": "start"})
                            
                            if is_collision:
                                col_x.append(path_arr[last_idx, 0])
                                col_y.append(path_arr[last_idx, 1])
                                markers_data.append({"frame": current_len + last_idx, "x": path_arr[last_idx, 0], "y": path_arr[last_idx, 1], "type": "collision"})
                            else:
                                if isinstance(true_goal, (list, tuple, np.ndarray)) and len(true_goal) >= 2:
                                    goals_x.append(true_goal[0])
                                    goals_y.append(true_goal[1])
                                    markers_data.append({"frame": current_len + last_idx, "x": true_goal[0], "y": true_goal[1], "type": "goal"})
                                else:
                                    goals_x.append(path_arr[last_idx, 0])
                                    goals_y.append(path_arr[last_idx, 1])
                                    markers_data.append({"frame": current_len + last_idx, "x": path_arr[last_idx, 0], "y": path_arr[last_idx, 1], "type": "goal"})
                                
                    all_x_by_agent[planner][k]["x"].extend(path_arr[:, 0])
                    all_x_by_agent[planner][k]["x"].append(np.nan) # Separator for seaborn
                    all_x_by_agent[planner][k]["y"].extend(path_arr[:, 1])
                    all_x_by_agent[planner][k]["y"].append(np.nan)
                    
        palette = get_color_palette()
        planners_list = list(planners)
        
        for planner, agents in all_x_by_agent.items():
            planner_idx = planners_list.index(planner) if planner in planners_list else 0
            planner_color = palette[planner_idx % len(palette)]
            
            for k, agent_data in enumerate(agents):
                if not agent_data["x"]:
                    continue
                    
                x_arr = np.array(agent_data["x"], dtype=float)
                y_arr = np.array(agent_data["y"], dtype=float)
                
                dists = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
                jumps = np.where((dists > 3.0) & ~np.isnan(dists))[0]
                split_indices = jumps + 1
                
                final_x = x_arr.copy()
                final_y = y_arr.copy()
                if len(split_indices) > 0:
                    final_x[split_indices] = np.nan
                    final_y[split_indices] = np.nan
                
                if len(agents) > 1:
                    opacity = 0.5
                    label = planner if k == 0 else None
                else:
                    opacity = 0.8
                    label = planner if k == 0 else None
                    
                line, = plt.plot(final_x, final_y, label=label, alpha=opacity, color=planner_color)
                lines_data.append((line, final_x, final_y))
                
        scat_starts = None
        scat_goals = None
        scat_cols = None
        
        if overlay_markers:
            if starts_x:
                scat_starts = plt.scatter(starts_x, starts_y, marker='o', color='#00bfb2', s=30, zorder=5)
            if goals_x:
                scat_goals = plt.scatter(goals_x, goals_y, marker='*', color='#ffc845', s=60, zorder=5)
            if col_x:
                scat_cols = plt.scatter(col_x, col_y, marker='x', color='#d3273e', s=50, linewidths=2, zorder=5)
            
        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
        plt.title(title)
        
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis('equal')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        
        if self.generate_gifs:
            markers_info = None
            if overlay_markers:
                markers_info = {
                    "data": markers_data,
                    "starts": scat_starts,
                    "goals": scat_goals,
                    "cols": scat_cols
                }
                
            gif_path = out_path.with_suffix(".gif")
            self._generate_gif(plt.gcf(), lines_data, markers_info, gif_path)
        
        plt.close()
