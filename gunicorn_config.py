import multiprocessing
import os

# Bind to the port provided by Render
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker configuration
workers = 1  # Use only 1 worker to minimize memory usage
worker_class = "sync"
threads = 2  # Use threads instead of multiple workers

# Timeout settings - increase to allow model loading
timeout = 120  # 2 minutes for worker timeout
graceful_timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app to load model once instead of per worker
preload_app = True

# Memory optimization
max_requests = 100  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 20
