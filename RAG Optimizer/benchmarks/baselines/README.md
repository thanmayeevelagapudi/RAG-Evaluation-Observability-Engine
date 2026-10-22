# Baseline Files

Place saved benchmark baselines in this folder.

Expected CI baseline path:

- `benchmarks/baselines/nfcorpus.json`

Generate or refresh baseline locally:

```bash
python -m benchmarks.run_benchmark --dataset nfcorpus --max-queries 50 --save-baseline
```

The CI benchmark gate only runs when `benchmarks/baselines/nfcorpus.json` exists in the repository.
