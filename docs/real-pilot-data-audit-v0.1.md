# Real sensor dataset: availability and limitations

The published pilot inventory describes 11 logger files identified as P001–P007
(with repeated runs for some identifiers). The recordings contain 21,887 samples
from three capacitive channels and two resistive channels at nominal 10 Hz.

Median aggregation produces 2,184 one-second summaries and 94 unlabeled
120-second windows with a 10-second stride. These recordings support descriptive
sensor analysis. They do not provide verified four-class risk labels.

## Available artifacts

- [Inventory and source hashes](../ai/data-manifests/real-pilot-v0.1.inventory.json)
- [Preparation configuration](../ai/configs/real-pilot-v0.1.json)
- [Session statistics](ai-evaluation/2026-09-09/tables/real_sessions.csv)
- [Channel statistics](ai-evaluation/2026-09-09/tables/real_channel_statistics.csv)
- [Original analysis](ai-evaluation/2026-09-09/laporan.md)

Raw CSV files are private. Published summaries do not imply permission to obtain
or redistribute the source recordings.

## Interpretation

Channel names are logger identifiers. They are not calibrated humidity,
volume, or clinically interpreted quantities. The four-strip physical design
has not been fully mapped to the three recorded capacitive channels here.

The descriptive preparation uses a provisional first-20-second baseline.
Its dry condition has not been independently verified. Normalized deltas are
for exploratory comparison and are not the raw-minus-baseline features used
by the bundled model.

P006 is a sensor-fault session. It is retained in the evidence but excluded from
the operational correlation summary. Correlations are calculated within sessions
and aggregated so that long recordings do not dominate.

Missing event annotations and risk labels prevent classifier training,
real-data accuracy estimation, or evaluation of warning lead time. Operator-
confirmed `Kap_7` requests may exercise the experimental adapter, but their
classes remain explicitly unvalidated.
