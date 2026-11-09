# G1 parity gap list

- benchmark: `/opt/arena_ws/data/benchmarks/parity-g1-roster-ped6`
- run ids: `parity-g1-roster-ped6`
- inventory keys: 267
- compared metric mappings: 24
- keys with a reviewed star mapping: 21
- keys without a scoped producer (port gaps): 216
- relocated series keys (not gaps): 30
- known remaining 1NF list violations: 12
- structural replacement keys (not parity keys): 5
- numeric star columns not claimed by any reviewed inventory mapping: 34

## Port gaps by owner

| owner | keys |
| --- | ---: |
| parity-g1 | 20 |
| port-eco | 102 |
| port-social | 94 |

## Port gaps by calculator

| calculator | keys |
| --- | ---: |
| edge_case_criticality | 24 |
| hhi_contact | 18 |
| social_forces | 17 |
| proxemics_extended | 16 |
| mechanism_metrics | 14 |
| path_metrics | 11 |
| edge_case_formation | 10 |
| task_replay | 9 |
| trajectory_naturalness | 8 |
| vln_metrics | 8 |
| acoustics | 7 |
| barn_score | 6 |
| compliance_metrics | 6 |
| regime_metrics | 6 |
| collision_metrics | 5 |
| condition_compliance | 5 |
| energy_extended | 5 |
| motion_metrics | 5 |
| acoustic_exposure | 4 |
| gaze_metrics | 4 |
| proxemics | 4 |
| semantic_interaction_metrics | 4 |
| energy | 3 |
| pedestrian_disturbance | 3 |
| time_metrics | 3 |
| clearance_metrics | 2 |
| holistic_metrics | 2 |
| mechanism_component | 2 |
| mutual_accommodation | 2 |
| characterization | 1 |
| mechanism_states | 1 |
| pedestrian_path_metrics | 1 |

## Port gaps by category

| category | keys |
| --- | ---: |
| social | 64 |
| ecological | 58 |
| performance | 50 |
| edge_case | 34 |
| naturalness | 8 |
| holistic | 2 |

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
