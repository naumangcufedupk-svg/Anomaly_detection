# Anomaly_detection
Anomaly Detection 
# Reproducible Anomaly Detection Benchmark

## Dataset
Numenta Anomaly Benchmark (NAB)

## Environment
pip install -r requirements.txt

## Run
python annomaly_detection_numan.py --download --max_files 50 --epochs 20 \
  --train_only_normal --label_radius 5 \
  --device cuda \
  --score_mode max_t \
  --point_agg max_overlap \
  --persistence_k 3 \
  --svm_nu 0.01 \
  --plot_curves

## Output
- results/baseline_results.csv
- results/ablation_results.csv
- results/metrics.json
- results/plots/
- results/failure_cases/
