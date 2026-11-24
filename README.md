# Rare Patterns in Recursive Generation

## Latest experiment

In a fresh autoregressive trigram study with 768 simulation worlds, selective archive reuse lowered mean sequence KL 10.2% versus tuned diffuse Bayesian resets. The rare-prefix primary test was inconclusive. Compared with whole-archive recall, selective reuse reduced rare-prefix excess KL by 37.0% on the mismatch cases.

[Read the report and replay the experiment](Selective%20archive%20reuse/README.md)

## Earlier experiment

Tests archive-aware Bayesian resets on 960 fresh simulated worlds.

When an archived source returned, the archive-aware reset reduced rare-context KL from 0.2331 to 0.1662 nats versus a tuned diffuse reset. The improvement was not established for recurrences absent from the archive.

[Full report, code and reproduction](Remembering%20real%20regimes/README.md)

[Earlier pilot](Recursive%20generation/README.md)
