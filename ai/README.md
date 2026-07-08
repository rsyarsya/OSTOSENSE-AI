# OSTOSENSE AI Contract Tools

This folder contains the dependency-free Tier 1 CSV contract and logger. It
does not contain model training or Tier 2 simulation.

## Minimal usage

```python
from ostosense_contract import (
    CapQuality,
    LigQuality,
    SampleRecord,
    Tier1CsvLogger,
)

logger = Tier1CsvLogger("tier1-data/session-batch-001")
logger.append_sample(
    SampleRecord.create(
        timestamp=1750000000000,
        session_id="session-001",
        capacitance_raw=12.5,
        lig_raw=410.0,
        cap_quality=CapQuality.OK,
        lig_quality=LigQuality.OK,
    )
)
```

`SampleRecord.create()` always derives `system_quality` from the two channel
qualities. Direct construction with an inconsistent aggregate is rejected.

Run tests from `ai/src`:

```powershell
python -m unittest discover -s ..\tests -v
```

The full field and event metadata definitions are in
`docs/ai-data-contract-v1.1.md`.
