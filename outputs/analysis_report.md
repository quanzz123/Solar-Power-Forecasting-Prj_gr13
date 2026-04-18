# Solar Power Analysis Report

## Generation Summary
```text
 plant_no  rows  timestamps  source_keys  ac_power_mean  dc_power_mean  zero_ac_ratio  total_daily_yield_sum  best_day_ac_power_total
        1 68778        3158           22     307.802752    3147.426211       0.464553           2.266901e+08            771576.161312
        2 67698        3259           22     241.277825     246.701961       0.526781           2.230575e+08            651437.736667
```

## Weather Summary
```text
 plant_no  rows  timestamps  ambient_temperature_mean  module_temperature_mean  irradiation_mean  irradiation_max
        1  3182        3182                 25.531606                31.091015          0.228313         1.221652
        2  3259        3259                 28.069400                32.772408          0.232737         1.098766
```

## Merged Summary
```text
 plant_no  merged_rows  missing_weather_rows       date_time_min       date_time_max  target_mean_ac_power
        1         3158                     1 2020-05-15 00:00:00 2020-06-17 23:45:00           6703.628149
        2         3259                     0 2020-05-15 00:00:00 2020-06-17 23:45:00           5011.974903
```

## Model Metrics
```text
  scope        MAE        RMSE       R2
overall 501.153261 1397.339167 0.958760
plant_1 257.740054  531.902718 0.995483
plant_2 743.809352 1901.841140 0.881824
```