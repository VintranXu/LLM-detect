#!/bin/bash

# 设置本地llama2-7b-hf模型路径
# export LOCAL_LLAMA_PATH="llama-2-7b-hf"  # 本地模型路径
# export LOCAL_LLAMA_PATH="/home/kemove/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased/"

model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=6

batch_size=1
d_model=768
d_ff=128

comment='TimeLLM-Weather-Local'

words=(
  fouling scaling blockage degradation cavitation 
  thermal_efficiency heat_transfer condensation deaeration 
  pressure_drop flow_restriction temperature_rise cooling_efficiency 
  pump_curve head_loss power_consumption mechanical_efficiency 
  control_deviation system_stability operational_margin safety_parameter 
  primary_loop secondary_loop steam_generator condenser deaerator 
  feedwater_system high_pressure_heater thermal_balance energy_conversion
  residual_analysis anomaly_detection fault_signature deviation_pattern
  baseline_drift measurement_error sensor_drift calibration_offset
  fault_propagation cascade_failure root_cause correlation_analysis
  threshold_violation alarm_logic diagnostic_reasoning failure_mode
  sensor_redundancy measurement_uncertainty signal_validation data_quality
  temperature_sensor pressure_transmitter flow_meter level_indicator
  instrumentation_fault sensor_bias measurement_drift signal_noise
  cross_validation sensor_correlation redundant_measurement backup_sensor
  feedwater_pump condensate_pump extraction_pump booster_pump
  moisture_separator_reheater low_pressure_heater extraction_steam
  condensate_polishing steam_dump_valve bypass_valve control_valve
  turbine_extraction steam_cycle efficiency_optimization heat_recovery
  residual_1 residual_2 residual_3 residual_4 residual_5 residual_6 residual_7 
  residual_8 residual_9 residual_10 residual_11 
  acceleration  deceleration  ramping
  sudden  gradual  progressive  exponential
  monotonic  non-monotonic  reversible
  increasing  decreasing  stationary
  convergent  divergent  stable  unstable
  noise  clean  filtered  smoothed
  corrupted  distorted  aliased
  resolution  precision  accuracy
  missing  incomplete  sparse  dense
  spike  burst  impulse  transient  surge
  dropout  dip  valley  trough  depression
  plateau  saturation  clipping  limiting
  instability  erratic  chaotic  irregular
  intermittent  sporadic  random  stochastic
)


# 多GPU模式
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 run_CLS.py \
  --task_name classification \
  --is_classification 1 \
  --is_training 1 \
  --root_path ./dataset \
  --data_path processed_data \
  --model_id fault_cls \
  --model $model_name \
  --data fault_cls \
  --num_classes 15 \
  --features M \
  --seq_len 2800 \
  --patch_len 32 \
  --stride 16 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --llm_model LLAMA \
  --model_comment $comment \
  # --custom_prototypes "${words[@]}"



