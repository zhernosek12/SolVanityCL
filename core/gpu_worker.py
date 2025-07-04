import os
import base58
import time
from loguru import logger

import multiprocessing

from typing import List, Optional, Tuple
from core.config import HostSetting
from base58 import b58encode

from core.searcher import Searcher
from core.postgres import Postgres

from dotenv import load_dotenv

from config import DB_CONFIG

load_dotenv()


def multi_gpu_worker(
    index: int,
    setting: HostSetting,
    task_queue: multiprocessing.Queue,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
):
    try:
        postgres = Postgres(
            db_config=DB_CONFIG
        )

        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )

        while True:
            task_batch = task_queue.get()
            if task_batch is None:
                break

            logger.info(f"Batch: {[f'{f[1]}__{f[2]}' for f in task_batch]}")

            # Convert to (prefix, suffix) -> row_id
            task_map = {}
            active_pairs = []
            for row_id, prefix, suffix, case_sensitive in task_batch:
                key = (row_id, prefix, suffix)
                task_map[key] = (row_id, case_sensitive)
                active_pairs.append(key)

            while active_pairs:
                logger.info(f"🔁 Left find pairs: {len(active_pairs)}")

                prefix_suffix_pairs = active_pairs
                case_sensitive = any(
                    task_map[pair][1] for pair in prefix_suffix_pairs)  # True если хоть одна пара чувствительна

                searcher.set_search_params_batch(prefix_suffix_pairs, case_sensitive)

                i = 0
                st = time.time()
                while True:
                    if i == 0:
                        logger.info(f"active_pairs: {active_pairs}")

                    result = searcher.find(i == 0)
                    found_something = False

                    if result:
                        results = get_results([r for r in result])

                        logger.info(f"finde: {results}")

                        for data in results:
                            address, private_key = data
                            found_pair = None

                            for row_id, prefix, suffix in active_pairs:
                                if address.startswith(prefix) and address.endswith(suffix):
                                    found_pair = (row_id, prefix, suffix)
                                    break

                            if found_pair is None:
                                continue

                            row_id, _ = task_map[found_pair]
                            logger.info(f"FOUND = Address: {address} => row_id: {row_id}")

                            postgres.update(**{
                                'row_id': row_id,
                                'start_address': address,
                                'private_address': private_key,
                                'status': 'success'
                            })

                            active_pairs.remove(found_pair)
                            found_something = True
                            break

                        if found_something:
                            break
                        else:
                            time.sleep(0.01)  # предотвращение tight loop

                    if time.time() - st > 1:
                        i = 0
                        st = time.time()
                    else:
                        i += 1

            logger.success("All pairs found!")

        postgres.close()

    except Exception as e:
        logger.exception(e)


def get_results(outputs):
    results = []
    for output in outputs:
        if not output[0]:
            continue
        seed_list = output[1]  # это list[int], длиной 32
        private_key_bytes = bytes(seed_list)

        expected_pubkey = private_key_bytes[32:]
        public_key_bytes = expected_pubkey

        solana_address = base58.b58encode(public_key_bytes).decode()
        private_key = b58encode(private_key_bytes).decode()

        results.append([solana_address, private_key])

    return results

