# SIN-Polarity

A tool for determining P-wave first motion polarity based on seismic waveform data.

## Overview

This project implements a workflow for:
1. Fetching seismic waveform data from IRIS
2. Picking P-wave arrival times (PhaseNet or STA/LTA)
3. Determining first motion polarity using template-based cross-correlation
4. (Optional) Focal mechanism inversion

## Examples

### Example 1: Oklahoma Earthquake (2014-10-10, mb4.5)

- **P-wave picking**: PhaseNet (Zhu and Beroza, 2019)
- **Bandpass filter**: 0.5–15 Hz
- **Distance range**: 0–1 degree

### Example 2: Myanmar Earthquake (2025-03-28, mw7.7)

- **P-wave picking**: STA/LTA (Allen, 1978)
- **Bandpass filter**: 0.01–0.2 Hz
- **Distance range**: 0–1 degree
- **Additional**: Focal mechanism inversion

## Workflow

```
a_tool_FetchDataFromIRIS.py   # Step 1: Download data from IRIS
        ↓
b_tool_phasenet.py            # Step 2a: Pick P-wave with PhaseNet
  or b_tool_StaLta.py         # Step 2b: Pick P-wave with STA/LTA
        ↓
c_tool_SIN-Polarity.py        # Step 3: Determine polarity
        ↓
FocalMechanismInversion.py    # Step 4 (optional): Focal mechanism
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

## Usage

1. Edit `a_tool_FetchDataFromIRIS.py` to set event parameters (origin time, location, distance range)
2. Run the scripts in sequence:
```bash
python a_tool_FetchDataFromIRIS.py
python b_tool_phasenet.py    # or b_tool_StaLta.py
python c_tool_SIN-Polarity.py
```

## Output

- `templates.csv`: P-wave arrival times for each station
- `StationinformationPhasenet.csv` / `StationinformationSTALTA.csv`: Polarity results (U=Up, D=Down)
- `output_FMsln.csv`: Focal mechanism solutions (if applicable)

## References

- PhaseNet: Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 216(1), 261-273.
- STA/LTA: Allen, R. V. (1978). Automatic earthquake recognition and timing from single traces. Bulletin of the Seismological Society of America, 68(5), 1521-1532.
