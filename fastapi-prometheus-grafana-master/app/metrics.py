"""Prometheus metrics for ML observability."""

from prometheus_client import Counter, Gauge, Histogram

prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["model_name"]
)

inference_time_histogram = Histogram(
    "ml_inference_duration_ms",
    "Model inference duration in milliseconds",
    ["model_name"],
    buckets=(10, 25, 50, 100, 200, 500, 1000, 2000, 5000)
)

fake_probability_histogram = Histogram(
    "ml_fake_probability",
    "Distribution of fake probabilities",
    ["model_name"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

avg_fake_probability_gauge = Gauge(
    "ml_avg_fake_probability",
    "Average fake probability of recent predictions",
    ["model_name"]
)

high_fake_probability_counter = Counter(
    "ml_high_fake_probability_total",
    "Count of high fake probability detections (>= 0.5)",
    ["model_name"]
)
