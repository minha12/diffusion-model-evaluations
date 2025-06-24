python scripts/evaluation/generate_report.py --metrics-dirs \
  ./results/bcss-1k-seed-23/metrics \
  ./results/segpath-1k-seed-23/metrics \
  ./results/drsk-1k-seed-23/metrics \
  --select-models sd35_controlnet \
  --output-dir ./results/cross_dataset_evaluation