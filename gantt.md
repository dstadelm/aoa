``` plantuml
@startgantt
project starts at 2024-02-05
sunday are closed
saturday are closed
printscale weekly
[Feasability Study/Architecture] as [TASK_1] requires 50 days
[HW Define I/O requirements] as [TASK_2] requires 7 days and starts 2024-03-01
[HW Schema] as [TASK_3] requires 25 days
[HW Layout] as [TASK_4] requires 25 days
[HW Production] as [TASK_5] requires 38 days
[HW Bringup] as [TASK_6] requires 13 days
[Setup Development Environment] as [TASK_7] requires 15 days
[FW Testing Infrastructure] as [TASK_8] requires 15 days
[FW Linux configuration] as [TASK_9] requires 8 days
[FW Architecture Infrastructure (PCIe)] as [TASK_10] requires 8 days
[FW Implementation Data Transfer and Signaling] as [TASK_11] requires 22 days
[FW Architecture RadarChain (AIE-ML)] as [TASK_12] requires 7 days
[FW Implementation RadarChain (AIE-ML)] as [TASK_13] requires 25 days
[SW Scene Generator Architecture] as [TASK_14] requires 8 days
[SW Scene Generator Implementation] as [TASK_15] requires 22 days
[SW Data Push Architecture] as [TASK_16] requires 3 days
[SW Data Push Implementation] as [TASK_17] requires 8 days
[SW Data Pop Architecture ] as [TASK_18] requires 3 days
[SW Data Pop Implementation] as [TASK_19] requires 8 days
[SW Gui Select Framework] as [TASK_20] requires 5 days
[SW Gui Pilot Performance Test] as [TASK_21] requires 2 days
[SW Gui Wireframe] as [TASK_22] requires 5 days
[SW Gui Implementation] as [TASK_23] requires 15 days
[Integration] as [TASK_24] requires 7 days
[TASK_2]->[TASK_3]
[TASK_3]->[TASK_4]
[TASK_4]->[TASK_5]
[TASK_5]->[TASK_6]
[TASK_1]->[TASK_7]
[TASK_1]->[TASK_8]
[TASK_8]->[TASK_9]
[TASK_7]->[TASK_9]
[TASK_9]->[TASK_10]
[TASK_10]->[TASK_11]
[TASK_8]->[TASK_12]
[TASK_7]->[TASK_12]
[TASK_12]->[TASK_13]
[TASK_8]->[TASK_14]
[TASK_7]->[TASK_14]
[TASK_14]->[TASK_15]
[TASK_15]->[TASK_16]
[TASK_16]->[TASK_17]
[TASK_17]->[TASK_18]
[TASK_18]->[TASK_19]
[TASK_8]->[TASK_20]
[TASK_7]->[TASK_20]
[TASK_20]->[TASK_21]
[TASK_21]->[TASK_22]
[TASK_22]->[TASK_23]
[TASK_19]->[TASK_24]
[TASK_13]->[TASK_24]
[TASK_23]->[TASK_24]
[MS 010] happens at 2024-03-16
[MS 020] happens at 2024-03-16
[MS 030] happens at 2024-04-08
[MS 040] happens at 2024-04-08
[MS 050] happens at 2024-04-11
[MS 060] happens at 2024-04-20
[MS 070] happens at 2024-04-30
[MS 080] happens at 2024-05-18
[MS 090] happens at 2024-05-31
[MS 100] happens at 2024-07-24
[MS 110] happens at 2024-09-04
[MS 120] happens at 2024-09-21
legend
[MS 010] Checklist of functional and performance requirements
[MS 020] Acquisition iWave eval kit
[MS 030] Feasability Analysis
[MS 040] GUI Wireframe Draft
[MS 050] GUI Draft Acknoledgement
[MS 060] GUI Performance Test
[MS 070] PCIe Example design on iWave eval board
[MS 080] HW Production Data
[MS 090] Simple Benchmark example on iWave eval Kit
[MS 100] 80% Functionality on iWave SOM
[MS 110] Production Release
[MS 120] Final Release
end legend
@endgantt
```
