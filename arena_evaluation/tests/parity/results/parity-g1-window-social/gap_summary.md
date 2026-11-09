# G1 parity gap list

- benchmark: `/opt/arena_ws/data/benchmarks/parity-g1-window-social`
- run ids: `parity-g1-window-social`
- inventory keys: 267
- compared metric mappings: 82
- keys with a reviewed star mapping: 79
- keys without a reviewed mapping but with a produced star column (UNMAPPED, manifest review): 125
- keys without a produced star column (NO_PRODUCER, real porting gaps): 26
- relocated series keys (not gaps): 37
- known remaining 1NF list violations: 12
- structural replacement keys (not parity keys): 5
- numeric star columns not claimed by any reviewed inventory mapping: 138
- minimum non-null comparisons per key: 3

## UNMAPPED (star column exists, manifest review needed)

| calculator | keys |
| --- | ---: |
| edge_case_criticality | 20 |
| hhi_contact | 17 |
| social_forces | 14 |
| proxemics_extended | 13 |
| mechanism_metrics | 10 |
| edge_case_formation | 8 |
| task_replay | 7 |
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
| holistic_metrics | 1 |

## NO_PRODUCER port gaps by owner

| owner | keys |
| --- | ---: |
| parity-g1 | 4 |
| port-eco | 16 |
| port-social | 6 |

## Port gaps by calculator

| calculator | keys |
| --- | ---: |
| path_metrics | 6 |
| acoustic_exposure | 4 |
| edge_case_criticality | 4 |
| motion_metrics | 3 |
| collision_metrics | 2 |
| edge_case_formation | 2 |
| task_replay | 2 |
| mechanism_states | 1 |
| pedestrian_path_metrics | 1 |
| time_metrics | 1 |

## Port gaps by category

| category | keys |
| --- | ---: |
| performance | 15 |
| edge_case | 6 |
| ecological | 5 |

## Known remaining 1NF list violations

| metric_key | reason |
| --- | --- |
| velocity | per-sample arrays still carried in combined_metrics; the 1NF relocation is pending |
| acceleration | per-sample arrays still carried in combined_metrics; the 1NF relocation is pending |
| jerk | per-sample arrays still carried in combined_metrics; the 1NF relocation is pending |
| path | trajectory arrays still carried in combined_metrics; the 1NF relocation is pending |
| path_odom | trajectory arrays still carried in combined_metrics; the 1NF relocation is pending |
| path_length_values | trajectory arrays still carried in combined_metrics; the 1NF relocation is pending |
| curvature | trajectory arrays still carried in combined_metrics; the 1NF relocation is pending |
| normalized_curvature | trajectory arrays still carried in combined_metrics; the 1NF relocation is pending |
| roughness | trajectory arrays still carried in combined_metrics; the 1NF relocation is pending |
| pedestrian_path | pedestrian trajectory array still carried in combined_metrics; the 1NF relocation is pending |
| collision_obstacles | collision provenance arrays still carried in combined_metrics; the 1NF relocation is pending |
| collisions | collision provenance arrays still carried in combined_metrics; the 1NF relocation is pending |

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
