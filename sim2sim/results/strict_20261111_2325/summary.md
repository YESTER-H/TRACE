Strict sim2sim sweep, 8 trials per scenario, 20 s each. stand = no fall; supine/prone = upright at the end (t = median get-up time); push = 2.4 m/s in 8 directions (falls -> recovered). err = mean |joint - dance ref| (rad) over the last 2 s of successful trials.

| group | condition | stand | supine (t) | prone (t) | push: falls -> recovered | err |
|---|---|---|---|---|---|---|
| baseline | nominal | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 4 -> 4 | 0.146 |
| baseline | nominal_no_noise | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 3 -> 3 | 0.146 |
| solver | euler | 8/8 | 8/8 (2.8s) | 8/8 (4.0s) | 4 -> 4 | 0.145 |
| solver | elliptic_impratio10 | 8/8 | 8/8 (2.9s) | 8/8 (4.1s) | 4 -> 4 | 0.146 |
| solver | cg_10iter | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 5 -> 5 | 0.146 |
| solver | dt_2ms | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 4 -> 4 | 0.146 |
| solver | dt_1ms | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 4 -> 4 | 0.146 |
| solver | soft_contact | 8/8 | 8/8 (2.8s) | 8/8 (4.0s) | 3 -> 3 | 0.146 |
| dynamics | friction_0.4 | 8/8 | 8/8 (2.8s) | 8/8 (3.9s) | 5 -> 5 | 0.145 |
| dynamics | friction_0.7 | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 5 -> 5 | 0.145 |
| dynamics | friction_1.5 | 8/8 | 8/8 (2.9s) | 8/8 (4.2s) | 4 -> 4 | 0.146 |
| dynamics | mass_x0.85 | 8/8 | 8/8 (2.8s) | 8/8 (3.9s) | 3 -> 3 | 0.141 |
| dynamics | mass_x1.15 | 8/8 | 8/8 (2.9s) | 8/8 (4.2s) | 8 -> 8 | 0.150 |
| dynamics | mass_rand_20pct | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 5 -> 5 | 0.146 |
| dynamics | torso_com_x+5cm | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 4 -> 4 | 0.144 |
| actuator | kp_kd_x0.8 | 8/8 | 8/8 (3.0s) | 8/8 (4.3s) | 8 -> 8 | 0.154 |
| actuator | kp_kd_x1.2 | 8/8 | 8/8 (2.8s) | 8/8 (3.9s) | 3 -> 3 | 0.141 |
| actuator | effort_x0.7 | 8/8 | 8/8 (2.9s) | 8/8 (4.4s) | 8 -> 8 | 0.146 |
| actuator | armature_x0 | 5/8 | 7/8 (2.8s) | 8/8 (4.0s) | 8 -> 7 | 0.144 |
| actuator | armature_x2 | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 5 -> 5 | 0.146 |
| actuator | joint_friction_0.3 | 8/8 | 8/8 (2.9s) | 8/8 (4.1s) | 4 -> 4 | 0.146 |
| actuator | joint_damping_0.5 | 8/8 | 8/8 (3.0s) | 8/8 (4.2s) | 5 -> 5 | 0.148 |
| actuator | explicit_pd | 8/8 | 8/8 (2.8s) | 8/8 (4.0s) | 4 -> 4 | 0.145 |
| timing | delay_5ms | 8/8 | 8/8 (3.0s) | 8/8 (4.1s) | 5 -> 5 | 0.147 |
| timing | delay_10ms | 8/8 | 8/8 (3.0s) | 8/8 (4.2s) | 7 -> 7 | 0.148 |
| timing | delay_20ms | 8/8 | 8/8 (3.0s) | 8/8 (4.4s) | 8 -> 7 | 0.151 |
| timing | delay_40ms | 4/8 | 3/8 (3.1s) | 3/8 (4.6s) | 8 -> 7 | 0.160 |
| timing | obs_noise_x2 | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 5 -> 5 | 0.143 |
| timing | gyro_bias_0.1 | 8/8 | 8/8 (2.9s) | 8/8 (4.0s) | 3 -> 3 | 0.144 |
| limit | limit_delay_60ms | 0/8 | 0/8 | 0/8 | 8 -> 0 | - |
| limit | limit_delay_100ms | 0/8 | 0/8 | 0/8 | 8 -> 0 | - |
| limit | limit_friction_0.2 | 8/8 | 8/8 (2.8s) | 2/8 (6.4s) | 8 -> 5 | 0.146 |
| limit | limit_kp_kd_x0.5 | 0/8 | 0/8 | 0/8 | 8 -> 0 | - |
| limit | limit_effort_x0.4 | 8/8 | 4/8 (3.1s) | 0/8 | 8 -> 4 | 0.146 |
| limit | limit_mass_x1.4 | 8/8 | 8/8 (3.1s) | 2/8 (8.4s) | 8 -> 5 | 0.160 |
| combined | random_mild | 8/8 | 8/8 (2.9s) | 8/8 (4.2s) | 7 -> 7 | 0.146 |
| combined | random_hard | 8/8 | 8/8 (3.0s) | 8/8 (4.3s) | 5 -> 5 | 0.148 |
