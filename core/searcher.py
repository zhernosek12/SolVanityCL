import logging
import time
import multiprocessing

from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

from core.config import HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_selected_gpu_devices,
)


class Searcher:
    def __init__(
        self,
        kernel_source: str,
        index: int,
        setting,
        chosen_devices: Optional[Tuple[int, List[int]]] = None,
    ):
        if chosen_devices is None:
            devices = get_all_gpu_devices()
        else:
            devices = get_selected_gpu_devices(*chosen_devices)

        enabled_device = devices[index]
        self.context = cl.Context([enabled_device])
        self.gpu_chunks = len(devices)
        self.command_queue = cl.CommandQueue(self.context)
        self.setting = setting
        self.index = index
        self.display_index = (
            index if chosen_devices is None else chosen_devices[1][index]
        )
        self.prev_time = None
        self.is_nvidia = "NVIDIA" in enabled_device.platform.name.upper()

        program = cl.Program(self.context, kernel_source).build()
        self.kernel = cl.Kernel(program, "generate_pubkey")

        self.memobj_key32 = None
        self.memobj_output = None
        self.memobj_occupied_bytes = None
        self.memobj_group_offset = None
        self.prefixes_buf = None
        self.prefix_lengths_buf = None
        self.suffix_buf = None
        self.output = None

    def set_search_params(self, PREFIXES: List[str], SUFFIX: str, CASE_SENSITIVE: bool):
        PREFIXES = [p.encode("utf-8") for p in PREFIXES]
        prefix_data = b''.join(PREFIXES)
        prefix_data_array = np.frombuffer(prefix_data, dtype=np.uint8)
        prefix_lengths = np.array([len(p) for p in PREFIXES], dtype=np.uint8)
        suffix_bytes = SUFFIX.encode("utf-8")
        suffix_data_array = np.frombuffer(suffix_bytes, dtype=np.uint8)
        suffix_len = np.uint32(len(suffix_data_array))
        case_sensitive = np.uint8(CASE_SENSITIVE)

        key32 = self.setting.key32
        occupied_bytes = np.array([self.setting.iteration_bytes], dtype=np.uint8)
        group_offset = np.array([self.index], dtype=np.uint8)

        if not hasattr(self, "memobj_key32") or self.memobj_key32 is None:
            self.memobj_key32 = cl.Buffer(
                self.context,
                cl.mem_flags.READ_ONLY,
                size=key32.nbytes
            )
        cl.enqueue_copy(self.command_queue, self.memobj_key32, key32)

        if not hasattr(self, "memobj_output") or self.memobj_output is None:
            self.output = np.zeros(33, dtype=np.ubyte)
            self.memobj_output = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE, size=self.output.nbytes
            )
        cl.enqueue_copy(self.command_queue, self.memobj_output, self.output)

        if not hasattr(self, "memobj_occupied_bytes") or self.memobj_occupied_bytes is None:
            self.memobj_occupied_bytes = cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE,
                size=occupied_bytes.nbytes
            )
        cl.enqueue_copy(self.command_queue, self.memobj_occupied_bytes, occupied_bytes)

        if not hasattr(self, "memobj_group_offset") or self.memobj_group_offset is None:
            self.memobj_group_offset = cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE,
                size=group_offset.nbytes
            )
        cl.enqueue_copy(self.command_queue, self.memobj_group_offset, group_offset)

        if not hasattr(self, "prefixes_buf") or self.prefixes_buf is None:
            self.prefixes_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=prefix_data_array.nbytes)
        elif self.prefixes_buf.size < prefix_data_array.nbytes:
            self.prefixes_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=prefix_data_array.nbytes)
        cl.enqueue_copy(self.command_queue, self.prefixes_buf, prefix_data_array)

        if not hasattr(self, "prefix_lengths_buf") or self.prefix_lengths_buf is None:
            self.prefix_lengths_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=prefix_lengths.nbytes)
        elif self.prefix_lengths_buf.size < prefix_lengths.nbytes:
            self.prefix_lengths_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=prefix_lengths.nbytes)
        cl.enqueue_copy(self.command_queue, self.prefix_lengths_buf, prefix_lengths)

        if not hasattr(self, "suffix_buf") or self.suffix_buf is None:
            self.suffix_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=max(1, suffix_data_array.nbytes))
        elif self.suffix_buf.size < suffix_data_array.nbytes:
            self.suffix_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=suffix_data_array.nbytes)
        cl.enqueue_copy(self.command_queue, self.suffix_buf,
                        suffix_data_array if suffix_len > 0 else np.array([0], dtype=np.uint8))

        self.output = np.zeros(33, dtype=np.ubyte)
        cl.enqueue_copy(self.command_queue, self.memobj_output, self.output)

        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output)
        self.kernel.set_arg(2, self.memobj_occupied_bytes)
        self.kernel.set_arg(3, self.memobj_group_offset)
        self.kernel.set_arg(4, self.prefixes_buf)
        self.kernel.set_arg(5, self.prefix_lengths_buf)
        self.kernel.set_arg(6, np.uint32(len(PREFIXES)))
        self.kernel.set_arg(7, self.suffix_buf)
        self.kernel.set_arg(8, suffix_len)
        self.kernel.set_arg(9, case_sensitive)

    def find(self, log_stats: bool = True) -> np.ndarray:
        start_time = time.time()
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
        )
        self.command_queue.flush()
        self.setting.increase_key32()
        if self.prev_time is not None and self.is_nvidia:
            time.sleep(self.prev_time * 0.98)
        cl.enqueue_copy(self.command_queue, self.output, self.memobj_output).wait()
        self.prev_time = time.time() - start_time
        if log_stats:
            logging.info(
                f"GPU {self.display_index} Speed: {global_worker_size / ((time.time() - start_time) * 1e6):.2f} MH/s"
            )
        return self.output


"""
def multi_gpu_init(
    index: int,
    setting: HostSetting,
    gpu_counts: int,
    stop_flag,
    lock,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
) -> List:
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )
        i = 0
        st = time.time()

        searcher.set_search_params(["pepe"], "", False)

        while True:
            result = searcher.find(i == 0)
            if result[0]:
                with lock:
                    if not stop_flag.value:
                        stop_flag.value = 1
                return result.tolist()
            if time.time() - st > max(gpu_counts, 1):
                i = 0
                st = time.time()
                with lock:
                    if stop_flag.value:
                        return result.tolist()
            else:
                i += 1
    except Exception as e:
        logging.exception(e)
    return [0]
"""


def multi_gpu_worker(
    index: int,
    setting: HostSetting,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
):
    try:
        path_save = "./"

        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )
        while True:
            task = task_queue.get()
            if task is None:
                break

            prefix, suffix, case_sensitive = task
            searcher.set_search_params([prefix], suffix, case_sensitive)  # Imortant == []

            i = 0
            st = time.time()
            while True:
                result = searcher.find(i == 0)
                if result[0]:
                    save_result([result.tolist()], path_save)
                    break
                if time.time() - st > 1:
                    i = 0
                    st = time.time()
                else:
                    i += 1
    except Exception as e:
        logging.exception(e)


def save_result(outputs: List, output_dir: str) -> int:
    from core.utils.crypto import save_keypair

    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        save_keypair(pv_bytes, output_dir)
    return result_count
