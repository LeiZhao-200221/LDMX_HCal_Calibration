# Reproducible container for the LDMX 2022 HCal test-beam analysis.
#
# build (from data_analysis/):  docker build -f env/Dockerfile -t ldmx-hcal-analysis .
# run:    docker run --rm -v /path/to/data:/data \
#                -v "$PWD/calibration:/app/calibration" -v "$PWD/results:/app/results" \
#                -v "$PWD/cache:/app/cache" \
#                ldmx-hcal-analysis ci --electron 0.3 0.5 -test
#
# Inside the container configure/config.yml is used with data_dir=/data (env/docker.yml is copied to
# configure/local.yml): mount the folder that holds root/, analysis_files/ (one sub-folder per run) and
# charge_injection_scans/ at /data (the generated csv go to /data/<stage>/ and /data/test/<stage>/; make-csv writes
# to /data/analysis_files/<run>/ and /data/test/analysis_files/<run>/). Mounted /app/calibration, /app/results and
# /app/cache hold calibration/<stage>/, results/fullset/<stage>/ and cache/<stage>/ (with -test: the test/ folder of
# each).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY env/requirements.txt env/requirements-lock.txt ./env/
RUN pip install --no-cache-dir -r env/requirements.txt -c env/requirements-lock.txt

COPY . .
RUN cp env/docker.yml configure/local.yml

ENTRYPOINT ["python", "src/runner.py"]
CMD ["--help"]
