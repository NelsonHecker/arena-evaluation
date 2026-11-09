from __future__ import annotations

import base64
import dataclasses
import math
import pathlib
import types
import typing

import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.metrics.ecological.mechanism_common import DoorGeom, ElevatorGeom, MechanismWorld, load_mechanism_world
from arena_evaluation.processing.pose_segments import teleport_jumps

if typing.TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

LEGACY_JUMP_M = 3.0
GIF_FRAMES = 100
GIF_FPS = 20
PED_TAIL_S = 5.0
SUCCESS_RESULTS = frozenset({"SUCCESS", "success", "reached", "goal_reached", "GOAL_REACHED"})
COLLISION_RESULTS = frozenset({"COLLISION", "collision", "CRASH"})
DOOR_COLORS = {"locked": "#d946ef", "closed": "#d3273e", "moving": "#f59e0b"}
CABIN_COLOR = "#06b6d4"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Frame:
    """Episode rows of one trajectory frame: one mapping per plotted episode."""

    rows: tuple[typing.Mapping[str, object], ...]
    columns: frozenset[str]

    @classmethod
    def of(cls, frame: pl.DataFrame) -> _Frame:
        return cls(rows=tuple(frame.to_dicts()), columns=frozenset(frame.columns))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def empty(self) -> bool:
        return not self.rows

    def has(self, name: str) -> bool:
        return name in self.columns

    def first(self, name: str, default: object = None) -> object:
        return self.rows[0].get(name, default) if self.rows else default

    def column(self, name: str) -> list[object]:
        return [row.get(name) for row in self.rows]

    def unique(self, name: str) -> list[object]:
        values: list[object] = []
        seen: set[str] = set()
        for row in self.rows:
            value = row.get(name)
            key = str(value)
            if key not in seen:
                seen.add(key)
                values.append(value)
        return values

    def where(self, name: str, value: object) -> _Frame:
        return _Frame(rows=tuple(row for row in self.rows if row.get(name) == value), columns=self.columns)

    def split_reference(self) -> tuple[_Frame, _Frame]:
        """(evaluated, reference) rows, using the ``is_reference`` column when the frame has it."""
        if not self.has("is_reference"):
            return self, _Frame(rows=(), columns=self.columns)
        evaluated = tuple(row for row in self.rows if not row.get("is_reference"))
        reference = tuple(row for row in self.rows if row.get("is_reference"))
        return _Frame(rows=evaluated, columns=self.columns), _Frame(rows=reference, columns=self.columns)

    def sorted_by(self, name: str) -> _Frame:
        return _Frame(rows=tuple(sorted(self.rows, key=lambda row: (row.get(name) is None, row.get(name)))), columns=self.columns)


def _sub_paths(path: object) -> list[np.ndarray]:
    """Path cell as a list of (N, >=2) arrays: one per agent for multi-agent cells, else one."""
    if not isinstance(path, (list, tuple, np.ndarray, pl.Series)) or len(path) == 0:
        return []
    first = path[0]
    nested = isinstance(first, (list, tuple, np.ndarray)) and len(first) > 0 and isinstance(first[0], (list, tuple, np.ndarray))
    groups = [p for p in path if p is not None and len(p) > 0] if nested else [path]
    out = []
    for group in groups:
        try:
            arr = np.array([list(p) for p in group], dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 2 and arr.shape[1] >= 2:
            out.append(arr)
    return out


def _row_time_ns(row: typing.Mapping[str, object], n: int) -> np.ndarray | None:
    """The row's `time` samples when they line up with its path, else None."""
    time_ns = row.get("time")
    if time_ns is None or len(time_ns) != n:  # type: ignore[arg-type]
        return None
    return np.asarray(time_ns, dtype=np.int64)


def _jump_indices(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None) -> np.ndarray:
    """Steps that are teleports (resets, elevator rides)."""
    if time_ns is not None:
        return teleport_jumps(x, y, time_ns)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.where(dists > LEGACY_JUMP_M)[0]


def _break_at_teleports(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Copies of x, y with NaN at every sample that lands a teleport, so lines stop there."""
    bx, by = x.astype(float).copy(), y.astype(float).copy()
    jumps = _jump_indices(x, y, time_ns)
    if len(jumps):
        bx[jumps + 1] = np.nan
        by[jumps + 1] = np.nan
    return bx, by


def _first_moving_index(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None) -> int:
    """Index where the first segment longer than 0.1 m starts (spawn jumps skipped)."""
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    jumps = _jump_indices(x, y, time_ns)
    starts = np.concatenate([[0], jumps + 1]).astype(int)
    ends = np.concatenate([jumps, [len(dists)]]).astype(int)
    for s_idx, e_idx in zip(starts, ends, strict=True):
        if e_idx > s_idx and np.sum(dists[s_idx:e_idx]) > 0.1:
            return int(s_idx)
    return int(starts[-1])


def _is_success(row: typing.Mapping[str, object]) -> bool:
    return row.get("result") in SUCCESS_RESULTS or bool(row.get("success"))


def _is_collision(row: typing.Mapping[str, object]) -> bool:
    return row.get("result") in COLLISION_RESULTS or (row.get("collision_amount") or 0) > 0


def _door_leaf(geom: DoorGeom | ElevatorGeom, progress: float) -> tuple[list[float], list[float]]:
    if isinstance(geom, ElevatorGeom):
        geom = DoorGeom(geom.name, geom.level, geom.door_start, geom.door_end, geom.activation_distance, geom.door_kind)
    if geom.kind != "sliding_split":
        (ax, ay), (bx, by) = geom.leaf(progress)
        return [ax, bx], [ay, by]
    (ax, ay), (ex, ey) = geom.start, geom.end
    f = (1.0 - progress) / 2.0
    return [ax, ax + (ex - ax) * f, math.nan, ex - (ex - ax) * f, ex], [ay, ay + (ey - ay) * f, math.nan, ey - (ey - ay) * f, ey]


def _door_color(progress: float, locked: bool) -> str:
    if locked:
        return DOOR_COLORS["locked"]
    return DOOR_COLORS["closed"] if progress < 0.02 else DOOR_COLORS["moving"]


def _state_geometry(world: MechanismWorld, state: typing.Mapping[str, object]) -> DoorGeom | ElevatorGeom | None:
    entity = f"{state['entity']}/{state['level']}" if state.get("level") else state["entity"]
    return world.door(str(entity)) if state.get("kind") == "door" else world.elevator(str(entity))


def _point_or_none(value: object) -> tuple[float, float] | None:
    """First two components of a (x, y, ...) pose, None when it is not a two-dimensional value."""
    if value is None or not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _flat_points(path_value: object) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(path_value, (list, tuple, np.ndarray)):
        return points
    nested = len(path_value) > 0 and isinstance(path_value[0], (list, tuple, np.ndarray)) and len(path_value[0]) > 0 and isinstance(path_value[0][0], (list, tuple, np.ndarray))
    groups = path_value if nested else [path_value]
    for group in groups:
        for point in group:
            flat = _point_or_none(point)
            if flat is not None:
                points.append(flat)
    return points


def _string_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


class TrajectoryRenderer(BasePlotRenderer):
    """Robot paths over the world map."""

    PLOT_TYPE = "trajectory"
    generate_gifs: bool = False
    _worlds: typing.ClassVar[dict[str, MechanismWorld | None]] = {}

    def _load_map_image(self, map_name: str, run_dir: pathlib.Path | None = None) -> dict | None:
        return MapRegistry.get_map(map_name, run_dir=run_dir)

    def _mechanism_world(self, map_name: str | None) -> MechanismWorld | None:
        if not map_name or not self.spec.options.get("overlay_mechanisms", True):
            return None
        if map_name not in self._worlds:
            try:
                self._worlds[map_name] = load_mechanism_world(map_name)
            except Exception as e:
                print(f"No mechanism geometry for {map_name}: {e}")
                self._worlds[map_name] = None
        return self._worlds[map_name]

    @staticmethod
    def _extract_start_goal(r: typing.Mapping[str, object]) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        s_pt = _point_or_none(r.get("start") or r.get("start_pos"))
        g_pt = _point_or_none(r.get("goal") or r.get("goal_pos"))
        path_val = r.get("path")
        if (s_pt is None or g_pt is None) and path_val is not None and len(path_val) > 0:  # type: ignore[arg-type]
            flat_pts = _flat_points(path_val)
            if flat_pts:
                if s_pt is None:
                    s_pt = flat_pts[0]
                if g_pt is None:
                    g_pt = flat_pts[-1]
        return s_pt, g_pt

    def _render_single_plot(self, frame: _Frame, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None = None) -> str:
        fig = go.Figure()
        diff_col = self.resolve_channel("color", fallback=self.spec.differentiate or "planner")
        map_meta = self._load_map_image(map_name, run_dir=run_dir) if self.spec.options.get("show_map", True) and map_name else None

        evaluated, _reference = frame.split_reference()
        planners = evaluated.unique(diff_col) if evaluated.has(diff_col) else ["unknown"]
        seen_planners: set[str] = set()
        overlay_markers = self.spec.options.get("overlay_markers", True)
        data_key = self.resolve_channel("x", fallback=self.spec.data_key or "path")

        for planner in planners:
            planner_frame = evaluated.where(diff_col, planner) if evaluated.has(diff_col) else evaluated
            if planner_frame.has("episode"):
                planner_frame = planner_frame.sorted_by("episode")
            all_x_by_agent: list[list[float | None]] = []
            all_y_by_agent: list[list[float | None]] = []
            markers: dict[str, list[list[float]]] = {key: [] for key in ("start_x", "start_y", "goal_x", "goal_y", "col_x", "col_y")}
            marker_indices: dict[str, list[list[int]]] = {"start_idx": [], "goal_idx": [], "col_idx": []}

            for row in planner_frame.rows:
                path = row.get(data_key)
                if path is None or isinstance(path, float) or len(path) == 0:  # type: ignore[arg-type]
                    continue
                for k, path_arr in enumerate(_sub_paths(path)):
                    time_ns = _row_time_ns(row, len(path_arr))
                    while len(all_x_by_agent) <= k:
                        all_x_by_agent.append([])
                        all_y_by_agent.append([])
                        if overlay_markers:
                            for values in markers.values():
                                values.append([])
                            for indices in marker_indices.values():
                                indices.append([])

                    if overlay_markers:
                        self._record_markers(markers, marker_indices, k, row, path_arr, time_ns, len(all_x_by_agent[k]))
                    if len(all_x_by_agent[k]) > 0:
                        all_x_by_agent[k].append(None)
                        all_y_by_agent[k].append(None)
                    bx, by = _break_at_teleports(path_arr[:, 0], path_arr[:, 1], time_ns)
                    all_x_by_agent[k].extend([None if np.isnan(v) else float(v) for v in bx])
                    all_y_by_agent[k].extend([None if np.isnan(v) else float(v) for v in by])

            palette = get_color_palette()
            planner_color = palette[planners.index(planner) % len(palette)]
            planner_name = str(planner)

            for k in range(len(all_x_by_agent)):
                if not all_x_by_agent[k]:
                    continue
                multi_agent = len(all_x_by_agent) > 1
                trace_name = f"{planner_name} - Agent {k}" if multi_agent else planner_name
                showlegend = planner_name not in seen_planners
                seen_planners.add(planner_name)
                fig.add_trace(
                    go.Scatter(
                        x=all_x_by_agent[k],
                        y=all_y_by_agent[k],
                        mode="lines",
                        name=trace_name,
                        legendgroup=planner_name,
                        showlegend=showlegend,
                        line=dict(color=planner_color),
                        opacity=0.5 if multi_agent else 0.8,
                        hovertemplate=f"<b>{planner_name}</b><br>{trace_name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                        customdata=list(range(len(all_x_by_agent[k]))),
                    )
                )
                if overlay_markers:
                    self._add_marker_traces(fig, planner_name, k, markers, marker_indices)

        self._add_theta_star_overlay(fig, evaluated, map_name, run_dir)
        self._add_pedestrian_overlay(fig, evaluated, run_dir)
        self._add_reference_overlay(fig, frame)

        world = self._mechanism_world(map_name or _string_or_none(frame.first("map")))
        if world is not None:
            for i, door in enumerate(world.doors):
                fig.add_trace(go.Scatter(x=[door.start[0], door.end[0]], y=[door.start[1], door.end[1]], mode="lines", name="Doors", legendgroup="mechanisms", showlegend=i == 0, line=dict(color=DOOR_COLORS["closed"], width=4), opacity=0.6, hovertemplate=f"<b>door {door.name}</b> ({door.kind})<extra></extra>"))
            for i, elevator in enumerate(world.elevators):
                xs = [corner[0] for corner in elevator.polygon] + [elevator.polygon[0][0]]
                ys = [corner[1] for corner in elevator.polygon] + [elevator.polygon[0][1]]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Elevator cabins", legendgroup="mechanisms", showlegend=i == 0, line=dict(color=CABIN_COLOR, width=1.5, dash="dot"), hovertemplate=f"<b>elevator {elevator.name}</b> level {elevator.level}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[elevator.door_start[0], elevator.door_end[0]], y=[elevator.door_start[1], elevator.door_end[1]], mode="lines", legendgroup="mechanisms", showlegend=False, line=dict(color=CABIN_COLOR, width=4), opacity=0.6, hoverinfo="skip"))

        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
        layout_args: dict[str, object] = dict(title=title, xaxis_title="X [m]", yaxis_title="Y [m]", yaxis=dict(scaleanchor="x", scaleratio=1), template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        if map_meta:
            layout_args = self._with_map_image(fig, map_meta, map_name, layout_args)
        fig.update_layout(layout_args)
        return self._inject_slider_js(fig)

    @staticmethod
    def _record_markers(markers: dict[str, list[list[float]]], indices: dict[str, list[list[int]]], agent: int, row: typing.Mapping[str, object], path_arr: np.ndarray, time_ns: np.ndarray | None, current_len: int) -> None:
        valid_mask = ~np.isnan(path_arr[:, 0])
        if not np.any(valid_mask):
            return
        clean_indices = np.where(valid_mask)[0]
        clean_t = time_ns[clean_indices] if time_ns is not None else None
        first_idx = int(clean_indices[_first_moving_index(path_arr[clean_indices, 0], path_arr[clean_indices, 1], clean_t)])
        last_idx = int(clean_indices[-1])
        markers["start_x"][agent].append(path_arr[first_idx, 0])
        markers["start_y"][agent].append(path_arr[first_idx, 1])
        indices["start_idx"][agent].append(current_len + first_idx)
        if _is_collision(row):
            markers["col_x"][agent].append(path_arr[last_idx, 0])
            markers["col_y"][agent].append(path_arr[last_idx, 1])
            indices["col_idx"][agent].append(current_len + last_idx)
        else:
            markers["goal_x"][agent].append(path_arr[last_idx, 0])
            markers["goal_y"][agent].append(path_arr[last_idx, 1])
            indices["goal_idx"][agent].append(current_len + last_idx)

    @staticmethod
    def _add_marker_traces(fig: go.Figure, planner_name: str, agent: int, markers: dict[str, list[list[float]]], indices: dict[str, list[list[int]]]) -> None:
        if markers["start_x"][agent]:
            fig.add_trace(
                go.Scatter(
                    x=markers["start_x"][agent],
                    y=markers["start_y"][agent],
                    mode="markers",
                    marker=dict(symbol="circle", size=8, color="#10b981", line=dict(width=1, color="#065f46")),
                    name=f"{planner_name} Starts",
                    legendgroup=planner_name,
                    showlegend=False,
                    opacity=0.9,
                    hovertemplate=f"<b>{planner_name}</b><br>Start<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                    customdata=indices["start_idx"][agent],
                )
            )
        if markers["goal_x"][agent]:
            fig.add_trace(
                go.Scatter(
                    x=markers["goal_x"][agent],
                    y=markers["goal_y"][agent],
                    mode="markers",
                    marker=dict(symbol="star", size=12, color="#ffc845"),
                    name=f"{planner_name} Goals",
                    legendgroup=planner_name,
                    showlegend=False,
                    opacity=0.9,
                    hovertemplate=f"<b>{planner_name}</b><br>Goal<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                    customdata=indices["goal_idx"][agent],
                )
            )
        if markers["col_x"][agent]:
            fig.add_trace(
                go.Scatter(
                    x=markers["col_x"][agent],
                    y=markers["col_y"][agent],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="#d3273e", line=dict(width=2)),
                    name=f"{planner_name} Collisions",
                    legendgroup=planner_name,
                    showlegend=False,
                    opacity=1.0,
                    hovertemplate=f"<b>{planner_name}</b><br>Collision<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                    customdata=indices["col_idx"][agent],
                )
            )

    @staticmethod
    def _with_map_image(fig: go.Figure, map_meta: dict, map_name: str | None, layout_args: dict[str, object]) -> dict[str, object]:
        try:
            with open(map_meta["png_path"], "rb") as file:
                encoded = base64.b64encode(file.read()).decode("utf-8")
            resolution = map_meta["resolution"]
            width_m = map_meta["width"] * resolution
            height_m = map_meta["height"] * resolution
            origin = map_meta.get("origin", [0.0, 0.0, 0.0])
            ox = float(origin[0])
            oy = float(origin[1])
            fig.add_layout_image(dict(source=f"data:image/png;base64,{encoded}", xref="x", yref="y", x=ox, y=oy + height_m, sizex=width_m, sizey=height_m, xanchor="left", yanchor="top", sizing="fill", opacity=0.45, layer="below"))
            layout_args["xaxis"] = dict(range=[ox, ox + width_m], title="X [m]")
            layout_args["yaxis"] = dict(range=[oy, oy + height_m], title="Y [m]", scaleanchor="x", scaleratio=1)
        except OSError as e:
            print(f"Failed to overlay map {map_name}: {e}")
        return layout_args

    def _add_theta_star_overlay(self, fig: go.Figure, evaluated: _Frame, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        if not (self.spec.options.get("overlay_theta_star", False) or self.spec.options.get("show_theta_star", False) or (self.spec.data_key and "theta" in self.spec.data_key)):
            return
        try:
            from arena_evaluation.processing.path.theta_star import compute_theta_star_for_episode
        except ImportError:
            return

        rendered = False
        seen_endpoints: set[tuple[str, float, float, float, float]] = set()
        for row in evaluated.rows:
            m_name = map_name or _string_or_none(row.get("map"))
            s_pt, g_pt = self._extract_start_goal(row)
            if not m_name or s_pt is None or g_pt is None:
                continue
            key = (str(m_name), round(s_pt[0], 2), round(s_pt[1], 2), round(g_pt[0], 2), round(g_pt[1], 2))
            if key in seen_endpoints:
                continue
            seen_endpoints.add(key)
            try:
                result = compute_theta_star_for_episode(str(m_name), s_pt, g_pt, run_dir=str(run_dir) if run_dir else None)
            except (OSError, ValueError):
                continue
            if result and result.success and len(result.path_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=result.path_x.tolist(),
                        y=result.path_y.tolist(),
                        mode="lines+markers",
                        name="Theta* Optimal Demonstration",
                        legendgroup="theta_star",
                        line=dict(color="#d97706", width=3, dash="dash"),
                        marker=dict(size=7, symbol="diamond", color="#d97706"),
                        opacity=0.95,
                        hovertemplate="<b>Theta* Optimal Path</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                        showlegend=not rendered,
                    )
                )
                rendered = True

    def _add_pedestrian_overlay(self, fig: go.Figure, evaluated: _Frame, run_dir: pathlib.Path | None) -> None:
        if not (self.spec.options.get("overlay_pedestrians", False) or self.spec.options.get("show_peds", False) or (self.spec.data_key and "ped" in self.spec.data_key)):
            return
        rendered = False
        for row in evaluated.rows:
            p_paths = row.get("pedestrian_path")
            if p_paths is None and run_dir is not None and row.get("episode") is not None:
                p_paths = _pedestrian_paths_from_topics(run_dir, int(row["episode"]))  # type: ignore[arg-type]
            if p_paths is None or not isinstance(p_paths, (list, tuple)):
                continue
            for p_idx, single_ped_path in enumerate(p_paths):
                if single_ped_path is None or len(single_ped_path) <= 1:
                    continue
                arr = np.array(single_ped_path)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=arr[:, 0].tolist(),
                        y=arr[:, 1].tolist(),
                        mode="lines",
                        name="Pedestrian Flows" if not rendered else f"Pedestrian {p_idx + 1}",
                        legendgroup="pedestrians",
                        line=dict(color="#8b5cf6", width=2, dash="dot"),
                        opacity=0.7,
                        hovertemplate=f"<b>Pedestrian {p_idx + 1}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                        showlegend=not rendered,
                    )
                )
                rendered = True

    def _add_reference_overlay(self, fig: go.Figure, frame: _Frame) -> None:
        if not (self.spec.options.get("overlay_reference", False) or (self.spec.data_key and "mar" in self.spec.data_key)):
            return
        rendered = False
        for row in frame.rows:
            is_ref = bool(row.get("is_reference")) if row.get("is_reference") is not None else False
            ref_type = str(row.get("reference_type", "")).lower()
            stage_name = str(row.get("stage", "")).lower()
            planner_name = str(row.get("planner", "")).lower()
            if "unhindered" in ref_type or "unhindered" in planner_name:
                continue
            if not ((ref_type == "unobstructed_robot") or ("unobstructed" in ref_type) or ("unobstructed" in stage_name) or (is_ref and ref_type != "unhindered_peds")):
                continue
            for r_arr in _sub_paths(row.get("path")):
                if np.any(np.abs(r_arr[:, 0]) > 500) or np.any(np.abs(r_arr[:, 1]) > 500):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=r_arr[:, 0].tolist(),
                        y=r_arr[:, 1].tolist(),
                        mode="lines",
                        name="Unobstructed Reference Robot",
                        legendgroup="unobstructed_ref",
                        line=dict(color="#06b6d4", width=3, dash="dashdot"),
                        opacity=0.9,
                        hovertemplate="<b>Unobstructed Reference</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                        showlegend=not rendered,
                    )
                )
                rendered = True

    def _inject_slider_js(self, fig: go.Figure) -> str:
        import uuid

        plot_id = f"traj_plot_{uuid.uuid4().hex}"
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=plot_id, config={"responsive": True})
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
                plotDiv._originalData = plotDiv.data.map(t => ({{x: t.x ? Array.from(t.x) : [], y: t.y ? Array.from(t.y) : []}}));
                var maxLen = 0;
                plotDiv.data.forEach(t => {{
                    if (t.customdata && t.customdata.length > 0) {{
                        let cdata = Array.isArray(t.customdata[0]) ? t.customdata.map(d => d[0]) : t.customdata;
                        let m = Math.max(...cdata);
                        if (m > maxLen) maxLen = m;
                    }} else if (t.x && t.x.length > maxLen) {{ maxLen = t.x.length; }}
                }});
                var slider = document.getElementById("slider_{plot_id}");
                var label = document.getElementById("val_{plot_id}");
                function formatTime(steps) {{
                    var totalMs = steps * 100;
                    var h = Math.floor(totalMs / 3600000);
                    var m = Math.floor((totalMs % 3600000) / 60000);
                    var s = Math.floor((totalMs % 60000) / 1000);
                    var ms = totalMs % 1000;
                    return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0') + ':' + String(ms).padStart(3, '0');
                }}
                if (maxLen > 0) {{
                    slider.max = maxLen;
                    slider.value = maxLen;
                    var maxStr = formatTime(maxLen);
                    label.innerText = maxStr + " / " + maxStr;
                    slider.addEventListener("input", function(e) {{
                        var limit = parseInt(e.target.value);
                        label.innerText = formatTime(limit) + " / " + maxStr;
                        plotDiv.data.forEach((trace, i) => {{
                            let isStaticRef = trace.name && (trace.name.includes("Theta*") || trace.name.includes("Unobstructed") || trace.legendgroup === "theta_star" || trace.legendgroup === "unobstructed_ref");
                            if (isStaticRef) {{
                                trace.x = plotDiv._originalData[i].x;
                                trace.y = plotDiv._originalData[i].y;
                            }} else if (trace.customdata && trace.customdata.length > 0) {{
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
        data_key = self.resolve_channel("x", fallback=self.spec.data_key or "path")
        if not data_key or df_filtered.schema.get(data_key) is None:
            return None

        run_dir = self.run_dir
        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]
            valid_groups = [column for column in group_cols if df_filtered.schema.get(column) is not None]
            if not valid_groups:
                return self._render_single_plot(_Frame.of(df_filtered), "", None, run_dir=run_dir)

            htmls: list[str] = []
            for name, group_df in df_filtered.group_by(valid_groups):
                group = _Frame.of(group_df)
                suffix = " | ".join(f"{column}: {value}" for column, value in zip(valid_groups, name, strict=True)) if isinstance(name, tuple) else f"{valid_groups[0]}: {name}"
                html = self._render_single_plot(group, suffix, _group_map_name(valid_groups, name, group), run_dir=run_dir)
                if html:
                    htmls.append(html)
            return htmls

        frame = _Frame.of(df_filtered)
        return self._render_single_plot(frame, "", _string_or_none(frame.first("map")), run_dir=run_dir)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        data_key = self.resolve_channel("x", fallback=self.spec.data_key or "path")
        if not data_key or df_filtered.schema.get(data_key) is None:
            return

        run_dir = self.run_dir
        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]
            valid_groups = [column for column in group_cols if df_filtered.schema.get(column) is not None]
            if not valid_groups:
                self.render_static_single(_Frame.of(df_filtered), out_path, "", None, run_dir)
                return

            for name, group_df in df_filtered.group_by(valid_groups):
                group = _Frame.of(group_df)
                if isinstance(name, tuple):
                    suffix = " | ".join(f"{column}: {value}" for column, value in zip(valid_groups, name, strict=True))
                    file_suffix = "_".join(str(value).replace(" ", "_").replace("/", "_") for value in name)
                else:
                    suffix = f"{valid_groups[0]}: {name}"
                    file_suffix = str(name).replace(" ", "_").replace("/", "_")
                group_out_path = out_path.with_name(f"{out_path.stem}_{file_suffix}{out_path.suffix}")
                self.render_static_single(group, group_out_path, suffix, _group_map_name(valid_groups, name, group), run_dir)
            return

        frame = _Frame.of(df_filtered)
        self.render_static_single(frame, out_path, "", _string_or_none(frame.first("map")), run_dir)

    def render_static_single(self, frame: _Frame, out_path: pathlib.Path, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        import matplotlib.pyplot as plt

        diff_col = self.resolve_channel("color", fallback=self.spec.differentiate or "planner")
        facet_col = self.spec.options.get("facet")
        if facet_col is not None and not frame.has(str(facet_col)):
            facet_col = None
        crop = bool(self.spec.options.get("crop", facet_col is not None))
        if map_name is None:
            map_name = _string_or_none(frame.first("map"))

        map_meta = self._load_map_image(map_name, run_dir=run_dir) if self.spec.options.get("show_map", True) and map_name else None
        world = self._mechanism_world(map_name)
        values = frame.unique(diff_col) if frame.has(diff_col) else ["unknown"]
        palette = get_color_palette()
        colors = {str(value): palette[index % len(palette)] for index, value in enumerate(values)}
        data_key = self.resolve_channel("x", fallback=self.spec.data_key or "path")

        if facet_col:
            groups = [(str(value), frame.where(str(facet_col), value)) for value in sorted(frame.unique(str(facet_col)), key=str)]
        else:
            groups = [(None, frame)]
        panels = [_Panel(name, group_frame, _tracks(group_frame, data_key, diff_col, colors)) for name, group_frame in groups]
        panels = [panel for panel in panels if panel.tracks]
        if not panels:
            return

        fig, axes = self._figure(plt, panels, map_meta, crop)
        for panel, ax in zip(panels, axes, strict=True):
            panel.ax = ax
            self._draw_map(ax, map_meta)
            self._draw_static(ax, panel, world, map_name, run_dir)
            if panel.name is not None:
                ax.set_title(panel.title(), fontsize=10)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")

        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
        if len(panels) > 1:
            fig.suptitle(title)
        else:
            axes[0].set_title(title)

        handles: list[object] = []
        labels: list[str] = []
        for ax in axes:
            handles_ax, labels_ax = ax.get_legend_handles_labels()
            handles += handles_ax
            labels += labels_ax
        by_label = dict(zip(labels, handles, strict=True))
        if len(panels) > 1:
            fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(len(by_label), 6), fontsize=9)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        else:
            axes[0].legend(by_label.values(), by_label.keys())
            fig.tight_layout()
        fig.savefig(out_path, dpi=300)

        if self.generate_gifs:
            self._generate_gif(fig, panels, out_path.with_suffix(".gif"))
        plt.close(fig)

    @staticmethod
    def _figure(plt: types.ModuleType, panels: list[_Panel], map_meta: dict | None, crop: bool) -> tuple[Figure, list[Axes]]:
        """One axes per panel, sized from the viewed area, and the shared view limits stored on each panel."""
        if map_meta and not crop:
            origin = map_meta.get("origin", [0.0, 0.0, 0.0])
            ox, oy = float(origin[0]), float(origin[1])
            x0, x1 = ox, ox + map_meta["width"] * map_meta["resolution"]
            y0, y1 = oy, oy + map_meta["height"] * map_meta["resolution"]
        else:
            points = np.vstack([point for panel in panels for point in panel.points()] or [np.zeros((1, 2))])
            x0, x1 = float(np.nanmin(points[:, 0])) - 2.0, float(np.nanmax(points[:, 0])) + 2.0
            y0, y1 = float(np.nanmin(points[:, 1])) - 2.0, float(np.nanmax(points[:, 1])) + 2.0
        for panel in panels:
            panel.limits = (x0, x1, y0, y1)
        n = len(panels)
        if n == 1:
            fig, ax = plt.subplots(figsize=(10, 10))
            return fig, [ax]
        width, height = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        ncols = n if height > 1.5 * width else (2 if width > 1.5 * height else 3)
        ncols = min(ncols, n)
        nrows = math.ceil(n / ncols)
        scale = min(16.0 / (ncols * width), 9.0 / (nrows * height))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width * scale + 1.0, nrows * height * scale + 1.5), squeeze=False)
        for ax in axes.flat[n:]:
            ax.axis("off")
        return fig, list(axes.flat[:n])

    @staticmethod
    def _draw_map(ax: Axes, map_meta: dict | None) -> None:
        if not map_meta:
            return
        import matplotlib.image as mpimg

        try:
            image = mpimg.imread(map_meta["png_path"])
            resolution = map_meta["resolution"]
            origin = map_meta.get("origin", [0.0, 0.0, 0.0])
            ox, oy = float(origin[0]), float(origin[1])
            ax.imshow(image, extent=(ox, ox + map_meta["width"] * resolution, oy, oy + map_meta["height"] * resolution), alpha=0.5, origin="upper", zorder=0)
        except (OSError, SyntaxError) as e:
            print(f"Failed to overlay map: {e}")

    def _draw_static(self, ax: Axes, panel: _Panel, world: MechanismWorld | None, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        x0, x1, y0, y1 = panel.limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        overlay_markers = self.spec.options.get("overlay_markers", True)

        if world is not None:
            self._draw_mechanisms(ax, panel, world)

        for track in panel.tracks:
            (track.line,) = ax.plot(track.x, track.y, label=track.label, alpha=track.alpha, color=track.color, zorder=4)

        if overlay_markers:
            starts = [track.start for track in panel.tracks if track.start is not None]
            goals = [track.end for track in panel.tracks if track.end is not None and not track.collision]
            collisions = [track.end for track in panel.tracks if track.end is not None and track.collision]
            if starts:
                ax.scatter([point[0] for point in starts], [point[1] for point in starts], marker="o", color="#00bfb2", s=30, zorder=5, label="Start")
            if goals:
                panel.goal_scatter = ax.scatter([point[0] for point in goals], [point[1] for point in goals], marker="*", color="#ffc845", s=60, zorder=5, label="Goal")
            if collisions:
                panel.collision_scatter = ax.scatter([point[0] for point in collisions], [point[1] for point in collisions], marker="x", color="#d3273e", s=50, linewidths=2, zorder=5, label="Collision")

        if self.spec.options.get("overlay_theta_star", False) or self.spec.options.get("show_theta_star", False) or (self.spec.data_key and "theta" in self.spec.data_key):
            self._draw_theta_star(ax, panel.frame, map_name, run_dir)

        if self.spec.options.get("overlay_pedestrians", False) or self.spec.options.get("show_peds", False) or (self.spec.data_key and "ped" in self.spec.data_key):
            first = True
            for track in panel.tracks:
                for ped in track.peds:
                    (ped.line,) = ax.plot(ped.xy[:, 0], ped.xy[:, 1], color="#8b5cf6", linestyle=":", linewidth=1.5, alpha=0.6, label="Pedestrians" if first else None, zorder=3)
                    (ped.dot,) = ax.plot([], [], "o", color="#8b5cf6", markersize=5, zorder=3)
                    first = False

        if self.spec.options.get("overlay_reference", False) or (self.spec.data_key and "mar" in self.spec.data_key):
            rendered = False
            for row in panel.frame.rows:
                if row.get("is_reference") and row.get("reference_type") == "unobstructed_robot":
                    for r_arr in _sub_paths(row.get("path")):
                        ax.plot(r_arr[:, 0], r_arr[:, 1], color="#06b6d4", linestyle="-.", linewidth=2.5, alpha=0.85, label="Unobstructed Reference" if not rendered else None, zorder=4)
                        rendered = True

    @staticmethod
    def _draw_theta_star(ax: Axes, frame: _Frame, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        try:
            from arena_evaluation.processing.path.theta_star import compute_theta_star_for_episode
        except ImportError:
            return
        for row in frame.rows:
            m_name = map_name or _string_or_none(row.get("map"))
            s_pt, g_pt = _point_or_none(row.get("start")), _point_or_none(row.get("goal"))
            if not m_name or s_pt is None or g_pt is None:
                continue
            try:
                result = compute_theta_star_for_episode(str(m_name), s_pt, g_pt, run_dir=str(run_dir) if run_dir else None)
            except (OSError, ValueError):
                continue
            if result and result.success and len(result.path_x) > 0:
                ax.plot(result.path_x, result.path_y, color="#d97706", linestyle="--", linewidth=2.5, marker="d", markersize=5, label="Theta* Reference", zorder=4)
                return

    @staticmethod
    def _draw_mechanisms(ax: Axes, panel: _Panel, world: MechanismWorld) -> None:
        """Doors as their closed slab and cabins as dotted boxes."""
        states = {(state["entity"], state.get("level")): state for state in panel.states()}
        for i, door in enumerate(world.doors):
            xs, ys = _door_leaf(door, 0.0)
            (line,) = ax.plot(xs, ys, color=DOOR_COLORS["closed"], linewidth=4, alpha=0.7, solid_capstyle="butt", label="Doors" if i == 0 else None, zorder=2)
            panel.leaves.append(_Leaf(door, line, states.get((door.name, door.level)) or states.get((door.name, None))))
        for i, elevator in enumerate(world.elevators):
            xs = [corner[0] for corner in elevator.polygon] + [elevator.polygon[0][0]]
            ys = [corner[1] for corner in elevator.polygon] + [elevator.polygon[0][1]]
            ax.plot(xs, ys, color=CABIN_COLOR, linewidth=1.2, linestyle=":", label="Elevator cabins" if i == 0 else None, zorder=2)
            lx, ly = _door_leaf(elevator, 0.0)
            (line,) = ax.plot(lx, ly, color=CABIN_COLOR, linewidth=4, alpha=0.7, solid_capstyle="butt", zorder=2)
            panel.leaves.append(_Leaf(elevator, line, states.get((elevator.name, elevator.level)) or states.get((elevator.name, None))))

    @staticmethod
    def _generate_gif(fig: Figure, panels: list[_Panel], out_path: pathlib.Path) -> None:
        """Replay every panel on one sim clock."""
        import matplotlib.animation as animation

        t_end = max((track.t[-1] for panel in panels for track in panel.tracks if len(track.t)), default=0.0)
        if t_end <= 0.0:
            return
        times = np.linspace(0.0, t_end, GIF_FRAMES + 1)
        clock = fig.text(0.01, 0.01, "", fontsize=11, family="monospace")

        def update(now: float) -> list:
            artists: list[Artist] = [clock]
            for panel in panels:
                goals, collisions = [], []
                for track in panel.tracks:
                    k = int(np.searchsorted(track.t, now, side="right"))
                    if track.line is not None:
                        track.line.set_data(track.x[:k], track.y[:k])
                        artists.append(track.line)
                    if track.end is not None and now >= track.t[-1]:
                        (collisions if track.collision else goals).append(track.end)
                    for ped in track.peds:
                        if ped.line is None or ped.dot is None:
                            continue
                        j = int(np.searchsorted(track.t, now - PED_TAIL_S, side="left"))
                        ped.line.set_data(ped.xy[j:k, 0], ped.xy[j:k, 1])
                        here = ped.xy[k - 1] if k > 0 else None
                        ped.dot.set_data(([here[0]], [here[1]]) if here is not None and np.isfinite(here[0]) else ([], []))
                        artists += [ped.line, ped.dot]
                for leaf in panel.leaves:
                    if leaf.state is None:
                        continue
                    k = min(int(np.searchsorted(leaf.t, now, side="right")), len(leaf.progress)) - 1
                    progress = float(leaf.progress[k]) if k >= 0 and np.isfinite(leaf.progress[k]) else 0.0
                    xs, ys = _door_leaf(leaf.geom, min(max(progress, 0.0), 1.0))
                    leaf.line.set_data(xs, ys)
                    leaf.line.set_color(_door_color(progress, bool(leaf.locked[k]) if k >= 0 else False))
                    artists.append(leaf.line)
                if panel.goal_scatter is not None:
                    panel.goal_scatter.set_offsets(np.array(goals) if goals else np.empty((0, 2)))
                    artists.append(panel.goal_scatter)
                if panel.collision_scatter is not None:
                    panel.collision_scatter.set_offsets(np.array(collisions) if collisions else np.empty((0, 2)))
                    artists.append(panel.collision_scatter)
            clock.set_text(f"t = {now:6.1f} s")
            return artists

        ani = animation.FuncAnimation(fig, update, frames=times, interval=1000 // GIF_FPS, blit=True)
        try:
            ani.save(out_path, writer="pillow", fps=GIF_FPS)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Failed to save GIF {out_path}: {e}")
        finally:
            clock.remove()


def _group_map_name(valid_groups: list[str], name: object, frame: _Frame) -> str | None:
    if "map" in valid_groups:
        index = valid_groups.index("map")
        return _string_or_none(name[index] if isinstance(name, tuple) else name)
    return _string_or_none(frame.first("map"))


def _pedestrian_paths_from_topics(run_dir: pathlib.Path, episode: int) -> list[list[tuple[float, float]]] | None:
    """Pedestrian polylines read from the episode's recorded telemetry when the frame does not carry them."""
    topics_dir = pathlib.Path(run_dir) / "episodes" / f"episode_{episode:03d}" / "topics"
    if not topics_dir.is_dir():
        return None
    from arena_evaluation.processing.parquet_store import TopicParquetStore

    bundles = TopicParquetStore.read(topics_dir)
    if not bundles:
        return None
    bundle = next(iter(bundles.values()))
    if bundle.pedestrian_tracks is None:
        return None
    tracks = bundle.pedestrian_tracks.collect() if isinstance(bundle.pedestrian_tracks, pl.LazyFrame) else bundle.pedestrian_tracks
    grouped = tracks.select(["ped_id", "x", "y"]).group_by("ped_id", maintain_order=True).agg(pl.concat_arr(["x", "y"]).alias("points"))
    return [[(float(x), float(y)) for x, y in points] for points in grouped["points"].to_list()]


class _Ped:
    def __init__(self, xy: np.ndarray) -> None:
        self.xy = xy
        self.line: Line2D | None = None
        self.dot: Line2D | None = None


class _Leaf:
    """A door slab artist bound to the recorded state series that moves it."""

    def __init__(self, geom: DoorGeom | ElevatorGeom, line: Line2D, state: typing.Mapping[str, object] | None) -> None:
        self.geom = geom
        self.line = line
        self.state = state
        if state is not None:
            self.t = np.asarray(state["t"], dtype=float)
            self.progress = np.asarray(state["progress"], dtype=float)
            self.locked = np.asarray(state["locked"], dtype=bool)


class _Track:
    """One episode path on a panel: broken at teleports, on its own relative sim clock."""

    def __init__(self, row: typing.Mapping[str, object], arr: np.ndarray, label: str | None, color: str, alpha: float) -> None:
        time_ns = _row_time_ns(row, len(arr))
        self.x, self.y = _break_at_teleports(arr[:, 0], arr[:, 1], time_ns)
        self.t = (time_ns - time_ns[0]) / 1e9 if time_ns is not None else np.arange(len(arr), dtype=float)
        self.label = label
        self.color = color
        self.alpha = alpha
        self.collision = _is_collision(row)
        valid = np.where(~np.isnan(arr[:, 0]))[0]
        self.start: tuple[float, float] | None = None
        self.end: tuple[float, float] | None = None
        if len(valid):
            first = int(valid[_first_moving_index(arr[valid, 0], arr[valid, 1], time_ns[valid] if time_ns is not None else None)])
            last = int(valid[-1])
            start_point = _point_or_none(row.get("start"))
            goal_point = _point_or_none(row.get("goal"))
            self.start = start_point if start_point is not None else (float(arr[first, 0]), float(arr[first, 1]))
            self.end = goal_point if (not self.collision and goal_point is not None) else (float(arr[last, 0]), float(arr[last, 1]))
        self.peds: list[_Ped] = []
        peds = row.get("pedestrian_path")
        for ped in peds if peds is not None else []:  # type: ignore[union-attr]
            if ped is not None and len(ped) == len(arr):
                xy = np.array([list(point[:2]) for point in ped], dtype=float)
                if np.isfinite(xy[:, 0]).any():
                    self.peds.append(_Ped(xy))
        self.states: list[dict] = []
        states = row.get("mechanism_states")
        for state in states if states is not None else []:  # type: ignore[union-attr]
            if len(state["progress"]) == len(arr):
                self.states.append({**state, "t": self.t})
        self.line: Line2D | None = None


class _Panel:
    def __init__(self, name: str | None, frame: _Frame, tracks: list[_Track]) -> None:
        self.name = name
        self.frame = frame
        self.tracks = tracks
        self.ax: Axes | None = None
        self.limits = (0.0, 1.0, 0.0, 1.0)
        self.leaves: list[_Leaf] = []
        self.goal_scatter: PathCollection | None = None
        self.collision_scatter: PathCollection | None = None

    def points(self) -> list[np.ndarray]:
        points = [np.column_stack([track.x, track.y]) for track in self.tracks]
        points += [ped.xy for track in self.tracks for ped in track.peds]
        return [point[np.isfinite(point[:, 0])] for point in points if np.isfinite(point[:, 0]).any()]

    def states(self) -> list[dict]:
        """Mechanism states to animate for the single episode on this panel, none when it overlays several."""
        with_states = [track for track in self.tracks if track.states]
        return with_states[0].states if len(with_states) == 1 else []

    def title(self) -> str:
        results = sorted({str(result) for result in self.frame.column("result") if result is not None})
        return f"{self.name}: {', '.join(results)}" if results else str(self.name)


def _tracks(frame: _Frame, data_key: str, diff_col: str, colors: dict[str, str]) -> list[_Track]:
    tracks: list[_Track] = []
    seen: set[str] = set()
    ordered = frame.sorted_by("episode") if frame.has("episode") else frame
    for row in ordered.rows:
        value = row.get(diff_col) if frame.has(diff_col) else "unknown"
        value_key = str(value)
        arrays = _sub_paths(row.get(data_key))
        for arr in arrays:
            label = None if value_key in seen else value_key
            seen.add(value_key)
            tracks.append(_Track(row, arr, label, colors.get(value_key, "#41b6e6"), 0.5 if len(arrays) > 1 else 0.8))
    return tracks
