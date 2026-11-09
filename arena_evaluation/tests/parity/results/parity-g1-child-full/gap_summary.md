# G1 parity gap list

- benchmark: `/opt/arena_ws/data/benchmarks/parity-g1-child-full`
- run ids: `parity-g1-child-full`
- inventory keys: 267
- compared metric mappings: 84
- keys with a reviewed star mapping: 81
- keys without a reviewed mapping but with a produced star column (UNMAPPED, manifest review): 134
- keys without a produced star column (NO_PRODUCER, real porting gaps): 4
- relocated series keys (not gaps): 48
- known remaining 1NF list violations: 0
- structural replacement keys (not parity keys): 5
- numeric star columns not claimed by any reviewed inventory mapping: 139
- minimum non-null comparisons per key: 3

## UNMAPPED (star column exists, manifest review needed)

| calculator | keys |
| --- | ---: |
| edge_case_criticality | 24 |
| hhi_contact | 17 |
| social_forces | 14 |
| proxemics_extended | 13 |
| edge_case_formation | 10 |
| mechanism_metrics | 10 |
| task_replay | 9 |
| vln_metrics | 7 |
| barn_score | 6 |
| condition_compliance | 5 |
| regime_metrics | 4 |
| pedestrian_disturbance | 3 |
| proxemics | 3 |
| gaze_metrics | 2 |
| mutual_accommodation | 2 |
| semantic_interaction_metrics | 2 |
| clearance_metrics | 1 |
| collision_metrics | 1 |
| holistic_metrics | 1 |

## NO_PRODUCER port gaps by owner

| owner | keys |
| --- | ---: |
| parity-g1 | 1 |
| port-eco | 3 |

## Port gaps by calculator

| calculator | keys |
| --- | ---: |
| acoustic_exposure | 4 |

## Port gaps by category

| category | keys |
| --- | ---: |
| ecological | 4 |

## Relocated series (no longer scalar columns)

| metric_key | reason |
| --- | --- |
| timeseries_char_time_s | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_power_total_w | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_power_mech_w | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_dba | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_vx_achieved | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_vy_achieved | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_wz_achieved | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_speed_achieved | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_accel_achieved | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_accel_target | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_efficiency | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_phase_kind | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_vx_target | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_vy_target | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_wz_target | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_speed_target | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_turn_radius_m | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_energy_intensity | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_energy_per_rad | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_char_leq_power | characterization per-sample arrays moved to the 1NF characterization series table; the legacy combined_metrics list is the pre-port mirror |
| timeseries_power_total_w | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_power_static_w | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_power_mechanical_w | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_power_thermal_w | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_battery_soc | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_velocity_linear | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_time_s | energy/power/velocity per-sample arrays moved to the 1NF robot series tables; the legacy combined_metrics list is the pre-port mirror |
| timeseries_acoustic_exposure_dba | acoustic exposure per-frame arrays moved to the 1NF pedestrian/acoustic series tables; no scalar fact column is intended |
| timeseries_acoustic_attenuation_db | acoustic exposure per-frame arrays moved to the 1NF pedestrian/acoustic series tables; no scalar fact column is intended |
| clearance_timeseries | clearance per-sample array moved to the 1NF robot series table; no scalar fact column is intended |
| timeseries_sfm_force | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| timeseries_ci | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| timeseries_min_ped_clearance | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| time_looking_at_pedestrians | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| time_looked_at_by_pedestrians | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| time_in_personal_space | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| hhi_stall_onsets_s | social per-sample/per-stall arrays moved to the 1NF series tables (social_forces sfm_force_n/ci, proxemics_extended min_ped_clearance_m, proxemics peds_in_personal_space, gaze looking_at_peds/looked_at_by_peds, hhi_contact stall onsets); no scalar fact column is intended |
| velocity | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| acceleration | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| jerk | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| path | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| path_odom | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| path_length_values | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| curvature | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| normalized_curvature | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| roughness | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| pedestrian_path | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
| time | path/motion/pedestrian-path per-sample arrays moved to the 1NF series tables: series_path (x_m, y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad, step_length_m, curvature_inv_m, normalized_curvature, roughness), series_motion (speed_mps, acceleration_mps2, jerk_mps3), series_pedestrian_path (ped_id, x_m, y_m, z_m, yaw_rad, vx_mps, vy_mps, speed_mps); `time` duplicated the sample clock every 1NF series table already carries as t_ns, so it is retired rather than relocated |
