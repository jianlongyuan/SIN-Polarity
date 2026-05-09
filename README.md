# SIN-Polarity

A tool for determining P-wave first motion polarity based on sine wave template.

## Overview

This project implements a workflow for:
1. Fetching seismic waveform data from IRIS
2. Picking P-wave arrival times (PhaseNet or STA/LTA)
3. Determining first motion polarity by calculating the cross-correlation values between P-wave and sine wave template

## Examples

### Example 1: Oklahoma Earthquake (2014-10-10, mb4.5)

### Example 2: Myanmar Earthquake (2025-03-28, mw7.7)

## Workflow

```
SIN_Polarity_Step1_Fetch_data.py                # Step 1: Download data from IRIS
        ↓
SIN-Polarity_Step2_First_arrival_picking.py     # Step 2: Pick P-wave first arrival by PhaseNet or STA/LTA
        ↓
SIN-Polarity_Step3_Polarity_identification.py   # Step 3: Determine P-wave first-motion polarity
```

## Quick Start

If you already have P-wave first arrival times from public datasets or other picking methods, you can directly use `SIN-Polarity_Step3_Polarity_identification.py` to determine P-wave first-motion polarity on the vertical component.

Prepare your arrival time data in `templates.csv` format with the following columns:
- `network`: Network code
- `station`: Station name
- `onsetP`: P-wave arrival time (seconds from trace start)

Example:
```
network,station,onsetP
GS,OK025,41.54
NX,STN02,41.9
```

## Requirements

- Python 3.x
- obspy
- numpy
- pandas
- scipy
- tensorflow (for PhaseNet)
- matplotlib
- numba (for focal mechanism inversion)

## Output

- `templates.csv`: P-wave arrival times for each station
- `StationinformationPhasenet.csv` / `StationinformationSTALTA.csv`: Polarity results (1='U', -1='D')

## Contact

Any questions or advices? Please contact:
- yuan_jianlong@126.com
- 1334631943@qq.com

