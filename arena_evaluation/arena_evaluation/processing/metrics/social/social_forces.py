from __future__ import annotations
import typing
import numpy as np

from ..base import BaseMetricCalculator

if typing.TYPE_CHECKING:
    from ....storage.schemas import AlignedEpisodeBundle


class SocialForcesCalculator(BaseMetricCalculator):
    """
    Social force metrics, computed on the NATIVE peds time base (full
    multi-rate standard, 2026-08-12). Robot pose is sampled onto the peds
    time axis by backward-asof join (100 ms tolerance).

    Metrics:
    - SFM (Helbing & Molnár 1995): cumulative / peak / mean repulsive force
      of pedestrians on the robot (A = 2.1 N, B = 0.3 m, cutoff 5 m,
      clamp 100 N). Distances are center-to-center, per the original model.
    - ESFM (Moussaïd et al. 2010): anisotropy-weighted SFM. In the original
      model the anisotropy w(phi) applies to the RECEIVER of the force; the
      headline variant therefore weights by the ROBOT's heading (the force
      the robot experiences — receiver-correct). The ped-heading variant
      (force the robot exerts on pedestrians) is reported separately as
      esfm_ped_*.
    - CI / SII: the Collision Index / Social Individual Index of the cited
      survey (Eq. 5) — a Gaussian personal-space violation index with
      sigma_px0 = sigma_py0 = 0.28 m (empirically set). The paper states the
      sigmas are equal "thus assuming that personal space is a perfect
      circle", which requires the squared (Gaussian) kernel; they may be
      adapted for cultures / relationships / contexts by changing the sigmas.

    Robot pose is ground truth (tf_gt) when recorded, odom otherwise.
    """

    NAME = "social_forces"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["odom", "peds"]

    UNITS = {
        "sfm_cumulative_force": "N·s",
        "sfm_peak_force": "N",
        "sfm_mean_force": "N",
        "esfm_cumulative_force": "N·s",
        "esfm_peak_force": "N",
        "esfm_mean_force": "N",
        "esfm_ped_cumulative_force": "N·s",
        "esfm_ped_peak_force": "N",
        "esfm_ped_mean_force": "N",
        "ci_max": "",
        "ci_mean": "",
        "timeseries_sfm_force": "N",
        "timeseries_ci": "",
    }

    PRIMARY_OUTPUTS = ["sfm_mean_force", "ci_mean"]
    OUTPUT_DIRECTIONS = {"sfm_mean_force": "lower", "ci_mean": "lower"}

    # SFM constants (Helbing & Molnár)
    _A = 2.1       # interaction strength (N)
    _B = 0.3       # interaction range (m)
    _PED_RADIUS = 0.3  # m
    _CUTOFF = 5.0  # m — only consider pedestrians within this radius
    _MAX_FORCE = 100.0  # N — clamp to prevent overflow when d_ij ≈ 0
    _LAMBDA_ESFM = 0.5  # anisotropy parameter (0.5 = moderate rear de-weighting)

    # CI / SII personal-space sigmas (paper Eq. 5, empirically 0.28 m)
    _SIGMA_PX0 = 0.28  # m
    _SIGMA_PY0 = 0.28  # m

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "sfm_cumulative_force",
            "sfm_peak_force",
            "sfm_mean_force",
            "esfm_cumulative_force",
            "esfm_peak_force",
            "esfm_mean_force",
            "esfm_ped_cumulative_force",
            "esfm_ped_peak_force",
            "esfm_ped_mean_force",
            "ci_max",
            "ci_mean",
            "timeseries_sfm_force",
            "timeseries_ci",
        ]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        # Guard: missing data
        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "peds_positions" not in peds_df.columns:
            return {k: None for k in self.output_keys()}
        pos_x, pos_y, yaw, t_odom = self.resolve_native_pose(episode)
        if len(pos_x) == 0:
            return {k: None for k in self.output_keys()}

        peds_time_ns = peds_df["time_ns"].to_numpy()
        peds_positions = peds_df["peds_positions"].to_list()
        num_peds_col = (
            peds_df["num_pedestrians"].to_numpy()
            if "num_pedestrians" in peds_df.columns
            else None
        )
        peds_headings_list = (
            peds_df["peds_headings"].to_list()
            if "peds_headings" in peds_df.columns
            else None
        )

        N = len(peds_time_ns)
        if N == 0:
            return {k: None for k in self.output_keys()}

        # Robot pose sampled onto the peds time axis (backward-asof, 100 ms)
        rpx, rpy, ryaw = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom)

        dt = np.diff(peds_time_ns) / 1e9
        dt = np.append(dt, 0.0)
        dt = np.where(dt == 0.0, 1e-6, dt)

        d_combined = self.robot_params.robot_radius + self._PED_RADIUS
        sx2 = 2.0 * self._SIGMA_PX0 ** 2
        sy2 = 2.0 * self._SIGMA_PY0 ** 2

        sfm_forces = []
        esfm_forces = []
        esfm_ped_forces = []
        ci_values = []

        for i in range(N):
            rx, ry = float(rpx[i]), float(rpy[i])
            peds_raw = peds_positions[i]
            peds_arr = self._parse_peds(
                peds_raw,
                num_peds_col[i] if num_peds_col is not None else None,
            )

            if peds_arr.shape[0] == 0:
                sfm_forces.append(0.0)
                esfm_forces.append(0.0)
                esfm_ped_forces.append(0.0)
                ci_values.append(0.0)
                continue

            # Pedestrian headings (may be missing → isotropic fallback)
            headings = None
            if peds_headings_list is not None and i < len(peds_headings_list):
                h_raw = peds_headings_list[i]
                if h_raw and len(h_raw) > 0:
                    if isinstance(h_raw, str):
                        import ast
                        try:
                            h_raw = ast.literal_eval(h_raw)
                        except Exception:
                            h_raw = []
                    headings = np.array(h_raw, dtype=np.float64) if len(h_raw) > 0 else None

            robot_yaw_i = float(ryaw[i])
            total_sfm = 0.0
            total_esfm = 0.0
            total_esfm_ped = 0.0
            max_ci = 0.0

            for j in range(peds_arr.shape[0]):
                px, py = peds_arr[j, 0], peds_arr[j, 1]
                dx = px - rx
                dy = py - ry
                d_ij = np.sqrt(dx**2 + dy**2)

                # ── CI / SII (paper Eq. 5, anisotropic Gaussian) ──
                # CI = max_i exp(-((xr-xpi)^2/(2 sx0^2) + (yr-ypi)^2/(2 sy0^2)))
                ci_val = np.exp(-((dx**2) / sx2 + (dy**2) / sy2))
                max_ci = max(max_ci, float(ci_val))

                # Skip SFM/ESFM beyond cutoff
                if d_ij > self._CUTOFF:
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    force_mag = self._A * np.exp((d_combined - d_ij) / self._B)
                force_mag = min(float(force_mag), self._MAX_FORCE)
                if np.isnan(force_mag) or np.isinf(force_mag):
                    force_mag = 0.0
                total_sfm += force_mag

                # ── ESFM — robot-heading anisotropy (receiver-correct) ──
                # phi = angle between robot facing and direction to the ped
                phi = robot_yaw_i - np.arctan2(py - ry, px - rx)
                w_robot = self._LAMBDA_ESFM + (1.0 - self._LAMBDA_ESFM) * (1.0 + np.cos(phi)) / 2.0
                w_robot = max(0.0, min(1.0, w_robot))
                total_esfm += force_mag * w_robot

                # ── ESFM — ped-heading anisotropy (force the robot exerts) ──
                w_ped = 1.0
                if headings is not None and j < len(headings):
                    ped_heading = float(headings[j])
                    phi_ped = ped_heading - np.arctan2(ry - py, rx - px)
                    w_ped = self._LAMBDA_ESFM + (1.0 - self._LAMBDA_ESFM) * (1.0 + np.cos(phi_ped)) / 2.0
                    w_ped = max(0.0, min(1.0, w_ped))
                total_esfm_ped += force_mag * w_ped

            sfm_forces.append(float(total_sfm))
            esfm_forces.append(float(total_esfm))
            esfm_ped_forces.append(float(total_esfm_ped))
            ci_values.append(float(max_ci))

        sfm_arr = np.array(sfm_forces)
        esfm_arr = np.array(esfm_forces)
        esfm_ped_arr = np.array(esfm_ped_forces)
        ci_arr = np.array(ci_values)

        with np.errstate(divide="ignore", invalid="ignore"):
            sfm_cumulative = float(np.sum(sfm_arr * dt))
            esfm_cumulative = float(np.sum(esfm_arr * dt))
            esfm_ped_cumulative = float(np.sum(esfm_ped_arr * dt))

        return {
            "sfm_cumulative_force": sfm_cumulative,
            "sfm_peak_force": float(np.max(sfm_arr)),
            "sfm_mean_force": float(np.mean(sfm_arr)),
            "esfm_cumulative_force": esfm_cumulative,
            "esfm_peak_force": float(np.max(esfm_arr)),
            "esfm_mean_force": float(np.mean(esfm_arr)),
            "esfm_ped_cumulative_force": esfm_ped_cumulative,
            "esfm_ped_peak_force": float(np.max(esfm_ped_arr)),
            "esfm_ped_mean_force": float(np.mean(esfm_ped_arr)),
            "ci_max": float(np.max(ci_arr)),
            "ci_mean": float(np.mean(ci_arr)),
            "timeseries_sfm_force": sfm_forces,
            "timeseries_ci": ci_values,
        }

    @staticmethod
    def _parse_peds(peds_raw, num_peds_hint=None):
        """Parse flat ped position list into (N, 2) or (N, 3) array."""
        if not peds_raw or len(peds_raw) == 0:
            return np.empty((0, 2))
        if isinstance(peds_raw, str):
            import ast
            try:
                peds_raw = ast.literal_eval(peds_raw)
            except Exception:
                return np.empty((0, 2))
        arr = np.array(peds_raw, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2))
        if arr.ndim == 1:
            if num_peds_hint and num_peds_hint > 0:
                if num_peds_hint * 3 == len(arr):
                    arr = arr.reshape(-1, 3)
                elif num_peds_hint * 2 == len(arr):
                    arr = arr.reshape(-1, 2)
            else:
                if len(arr) % 3 == 0:
                    arr = arr.reshape(-1, 3)
                elif len(arr) % 2 == 0:
                    arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.empty((0, 2))
        return arr
