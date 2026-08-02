from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class PedestrianDisturbanceCalculator(BaseMetricCalculator):
    """
    Calculates pedestrian disturbance metrics by comparing contestant pedestrian motion
    against baseline pedestrian motion (unhindered_peds reference or expected trajectory).

    Metrics:
    - ped_path_deflection_m: Mean spatial deflection (m) of pedestrians from baseline
    - ped_velocity_delay_ratio: Relative slowdown of pedestrians (0.0 = no slowdown)
    - ped_round_trips_completed: Estimated number of waypoint round-trips completed
    """

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
        actual_coords: (N, 2)
        reference_coords: (M, 2)
        """
        if len(actual_coords) == 0 or len(reference_coords) == 0:
            return 0.0
        
        diffs = actual_coords[:, np.newaxis, :] - reference_coords[np.newaxis, :, :]  # (N, M, 2)
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
            if not row or not isinstance(row, (list, tuple)):
                continue
            for agent_idx, (px, py) in enumerate(row):
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
            step_dists = np.sqrt(dx**2 + dy**2)
            total_dist = float(np.sum(step_dists))
            
            dists_from_start = np.sqrt(np.sum((arr - arr[0])**2, axis=1))
            max_disp = float(np.max(dists_from_start))
            
            if max_disp > 0.5:
                trips = total_dist / (2.0 * max_disp)
                round_trips += max(0, int(np.round(trips)))

        # Velocity delay: compare actual mean speed to baseline desired speed (~1.2 m/s)
        time_ns = episode.data["time_ns"].to_numpy() if "time_ns" in episode.data.columns else None
        avg_speed = 0.0
        if time_ns is not None and len(time_ns) > 1:
            dt_s = (time_ns[-1] - time_ns[0]) / 1e9
            if dt_s > 0:
                all_dists = [
                    np.sum(np.sqrt(np.sum(np.diff(np.array(pts), axis=0)**2, axis=1)))
                    for pts in agent_paths.values()
                    if len(pts) > 1
                ]
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
