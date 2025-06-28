import logging
import multiprocessing

from flask import Flask, request, jsonify
from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
)
from core.searcher import multi_gpu_worker
from core.utils.helpers import check_character, load_kernel_source

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
task_queue = None
result_queue = None
processes = []


def start_gpu_workers(setting: HostSetting, gpu_counts: int, chosen_devices=None):
    global task_queue, result_queue, processes

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    processes = []

    for index in range(gpu_counts):
        p = multiprocessing.Process(
            target=multi_gpu_worker,
            args=(
                index,
                setting,
                task_queue,
                result_queue,
                chosen_devices,
            ),
        )
        p.start()
        processes.append(p)

    return processes, task_queue, result_queue


def enqueue_task(task_queue, prefix: str, suffix: str, case_sensitive: bool):
    task_queue.put((prefix, suffix, case_sensitive))


@app.route("/enqueue", methods=["POST"])
def enqueue_route():
    try:
        data = request.get_json()
        starts_with = data.get("prefix", "")
        ends_with = data.get("suffix", "")
        case_sensitive = data.get("case_sensitive", False)

        for prefix in starts_with:
            check_character("starts_with", prefix)

        check_character("ends_with", ends_with)

        enqueue_task(task_queue, starts_with, ends_with, case_sensitive)

        return jsonify({"status": "task added"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def main():
    global processes, task_queue, result_queue

    kernel_source = load_kernel_source()
    setting = HostSetting(kernel_source, iteration_bits=DEFAULT_ITERATION_BITS)
    gpu_counts = len(get_all_gpu_devices())

    processes, task_queue, result_queue = start_gpu_workers(setting, gpu_counts, None)

    logging.info("GPU workers are running. API server is available at http://localhost:5000")

    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        for _ in processes:
            task_queue.put(None)
        for p in processes:
            p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()