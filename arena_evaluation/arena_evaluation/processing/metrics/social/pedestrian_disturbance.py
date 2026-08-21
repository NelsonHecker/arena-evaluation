from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class PedestrianDisturbanceCalculator(BaseMetricCalculator):
    """Computes pedestrian deflection and slowdown metrics against baseline trajectories."""

    NAME = "pedestrian_disturbance"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["peds"]

    UNITS = {
        "ped_path_deflection_m": "m",
        "ped_velocity_delay_ratio": "",
        "ped_round_trips_completed": "",
    }

    PRIMARY_OUTPUTS = ["ped_path_deflection_m"]

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_path_deflection_m",
            "ped_velocity_delay_ratio",
            "ped_round_trips_completed",
        ]

    @staticmethod
    def _compute_trajectory_deflection(
        actual_coords: np.ndarray, reference_coords: np.ndarray
    ) -> float:
        """
        Compute mean distance between actual trajectory points and nearest reference trajectory points.
        actual_coords: (N, 2) or (N, 3)
        reference_coords: (M, 2) or (M, 3)
        """
        if len(actual_coords) == 0 or len(reference_coords) == 0:
            return 0.0

        valid_act = actual_coords[~np.isnan(actual_coords).any(axis=1)]
        valid_ref = reference_coords[~np.isnan(reference_coords).any(axis=1)]
        if len(valid_act) == 0 or len(valid_ref) == 0:
            return 0.0

        valid_act = valid_act[:, :2]
        valid_ref = valid_ref[:, :2]

        diffs = valid_act[:, np.newaxis, :] - valid_ref[np.newaxis, :, :]  # (N, M, 2)
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))  # (N, M)
        min_dists = np.min(dists, axis=1)  # (N,)
        return float(np.mean(min_dists))

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        if episode.data is None or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}

        peds_pos = episode.data["peds_positions"].to_list()
        if not peds_pos or len(peds_pos) == 0:
            return {k: None for k in self.output_keys()}

        # Extract pedestrian positions per agent
        agent_paths: dict[int, list[tuple[float, float]]] = {}
        for row in peds_pos:
            arr = self._parse_peds(row)
            pts = [
                (float(item[0]), float(item[1]))
                for item in arr
                if not np.isnan(item[0]) and not np.isnan(item[1])
            ]

            for agent_idx, (px, py) in enumerate(pts):
                if agent_idx not in agent_paths:
                    agent_paths[agent_idx] = []
                agent_paths[agent_idx].append((px, py))

        if not agent_paths:
            return {k: None for k in self.output_keys()}

        round_trips = 0
        total_deflection = 0.0
        num_agents = len(agent_paths)

        for agent_idx, coords in agent_paths.items():
            arr = np.array(coords)
            if len(arr) < 5:
                continue

            dx = np.diff(arr[:, 0])
            dy = np.diff(arr[:, 1])
            step_dists = np.hypot(dx, dy)
            total_dist = float(np.sum(step_dists))
            
            diffs = arr - arr[0]
            dists_from_start = np.hypot(diffs[:, 0], diffs[:, 1])
            max_disp = float(np.max(dists_from_start))
            
            if not np.isnan(max_disp) and not np.isinf(max_disp) and max_disp > 0.5:
                # Avoid 2.0 * max_disp which can overflow if max_disp is near max float64
                trips = (total_dist / 2.0) / max_disp
                if not np.isnan(trips) and not np.isinf(trips):
                    round_trips += max(0, int(np.round(trips)))

            # Standalone geometric lateral deflection against straight-line start-to-end vector
            p_start, p_end = arr[0], arr[-1]
            seg_vec = p_end - p_start
            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
            if seg_len > 0.5:
                u_vec = seg_vec / seg_len
                # Cross-product magnitude gives perpendicular distance in 2D
                rel_pos = arr - p_start
                lateral_dists = np.abs(rel_pos[:, 0] * u_vec[1] - rel_pos[:, 1] * u_vec[0])
                total_deflection += float(np.mean(lateral_dists))

        # Velocity delay: compare actual mean speed to baseline desired speed (~1.2 m/s)
        time_ns = episode.data["time_ns"].to_numpy() if "time_ns" in episode.data.columns else None
        avg_speed = 0.0
        if time_ns is not None and len(time_ns) > 1:
            dt_s = (time_ns[-1] - time_ns[0]) / 1e9
            if dt_s > 0:
                all_dists = []
                for pts in agent_paths.values():
                    if len(pts) > 1:
                        p_arr = np.array(pts)
                        all_dists.append(np.sum(np.hypot(np.diff(p_arr[:, 0]), np.diff(p_arr[:, 1]))))
                if all_dists:
                    avg_dist = float(np.mean(all_dists))
                    avg_speed = avg_dist / dt_s

        desired_speed = 1.2  # Nominal pedsim desired speed
        velocity_delay_ratio = max(0.0, float(1.0 - (avg_speed / desired_speed))) if avg_speed > 0 else 0.0

        return {
            "ped_path_deflection_m": round(total_deflection / num_agents, 3) if num_agents > 0 else 0.0,
            "ped_velocity_delay_ratio": round(velocity_delay_ratio, 3),
            "ped_round_trips_completed": round_trips / num_agents if num_agents > 0 else 0,
        }
