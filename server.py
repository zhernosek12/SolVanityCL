import multiprocessing

from loguru import logger
from math import ceil

from core.postgres import Postgres
from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
)
from core.gpu_worker import multi_gpu_worker
from core.utils.helpers import check_character, load_kernel_source
from core.utils.parser import parse_wallet_pattern

from config import DB_CONFIG, SUBCHUNK_MAX

task_queues = None
gpu_counts = 0
processes = []


def start_gpu_workers(setting: HostSetting, chosen_devices=None):
    global task_queues, gpu_counts, processes

    task_queues = []
    processes = []

    for index in range(gpu_counts):
        queue = multiprocessing.Queue()
        task_queues.append(queue)

        p = multiprocessing.Process(
            target=multi_gpu_worker,
            args=(
                index,
                setting,
                queue,
                chosen_devices,
            ),
        )
        p.start()
        processes.append(p)

    return processes, task_queues


def event_new_row(rows):
    global gpu_counts
    global task_queues

    batch = []

    logger.info("Добавляем новую партию...")

    for row in rows:
        row_id, start_address = row
        wallet_start, wallet_end = parse_wallet_pattern(start_address)
        case_sensitive = True

        if wallet_start == '':
            wallet_start = "x"

        check_character("starts_with", wallet_start)
        check_character("ends_with", wallet_end)

        batch.append((row_id, wallet_start, wallet_end, case_sensitive))

    batch_size = len(batch)
    chunk_size = ceil(batch_size / gpu_counts)

    for i in range(gpu_counts):
        start = i * chunk_size
        end = min(start + chunk_size, batch_size)
        chunk = batch[start:end]
        if chunk:
            if len(chunk) > SUBCHUNK_MAX:
                for j in range(0, len(chunk), SUBCHUNK_MAX):
                    subchunk = chunk[j:j + SUBCHUNK_MAX]
                    task_queues[i].put(subchunk)
                    logger.info(f"Added to queue: {len(subchunk)}")
            else:
                task_queues[i].put(chunk)
                logger.info(f"Added to queue: {len(chunk)}")


def main():
    global processes, task_queues, gpu_counts

    kernel_source = load_kernel_source()

    setting = HostSetting(kernel_source, iteration_bits=DEFAULT_ITERATION_BITS)

    gpu_counts = len(get_all_gpu_devices())

    logger.info("Запускаем отработку задач...")

    processes, task_queues = start_gpu_workers(setting, None)

    logger.info("Инициализация базы...")

    postgres = Postgres(
        db_config=DB_CONFIG
    )

    logger.info("Запуск слушателя...")

    postgres.start_listen(event_new_row)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()