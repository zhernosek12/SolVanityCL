import time
import numpy as np
import pyopencl as cl

from loguru import logger
from typing import List, Optional, Tuple


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
        self.start_index = 0
        self.prefix_suffix_pairs = []
        self.prev_time = None
        self.is_nvidia = "NVIDIA" in enabled_device.platform.name.upper()

        program = cl.Program(self.context, kernel_source).build()
        self.kernel = cl.Kernel(program, "generate_pubkey")

        self.group_offset = None
        self.memobj_key32 = None
        self.memobj_output = None
        self.memobj_out_index = None
        self.memobj_occupied_bytes = None
        self.memobj_group_offset = None
        self.prefixes_buf = None
        self.prefix_lengths_buf = None
        self.suffix_buf = None

        self.suffixes_buf = None
        self.suffix_lengths_buf = None

        self.output = None
        self.output_index = None

    def set_search_params_batch(self, prefix_suffix_pairs: List[Tuple[str, str, str]], case_sensitive: bool):
        prefix_bytes_list = []
        suffix_bytes_list = []
        prefix_lengths = []
        suffix_lengths = []

        for _, prefix, suffix in prefix_suffix_pairs:
            prefix_b = prefix.encode('utf-8') if prefix else b''
            suffix_b = suffix.encode('utf-8') if suffix else b''

            prefix_bytes_list.append(prefix_b)
            suffix_bytes_list.append(suffix_b)

            prefix_lengths.append(len(prefix_b))
            suffix_lengths.append(len(suffix_b))

        self.prefix_suffix_pairs = prefix_suffix_pairs

        flat_prefixes = b''.join(prefix_bytes_list)
        flat_suffixes = b''.join(suffix_bytes_list)

        prefix_buf = np.frombuffer(flat_prefixes, dtype=np.uint8) if flat_prefixes else np.zeros(1, dtype=np.uint8)

        if flat_suffixes:
            suffix_buf = np.frombuffer(flat_suffixes, dtype=np.uint8)
        else:
            suffix_buf = np.zeros(1, dtype=np.uint8)

        prefix_lengths = np.array(prefix_lengths, dtype=np.uint8)
        suffix_lengths = np.array(suffix_lengths, dtype=np.uint8)
        case_sensitive = np.uint8(case_sensitive)
        pair_count = np.uint32(len(prefix_suffix_pairs))

        self.output = np.empty(65, dtype=np.uint8)
        self.output_index = np.zeros(1, dtype=np.uint32)

        occupied_bytes = np.array([self.setting.iteration_bytes], dtype=np.uint32)
        self.setting.key32 = self.setting.generate_key32()  # уникальный seed для уникального приватника
        self.group_offset = np.array([self.index], dtype=np.uint32)

        self.memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.setting.key32
        )

        self.memobj_output = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.output
        )

        self.memobj_occupied_bytes = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=occupied_bytes
        )

        self.memobj_group_offset = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.group_offset,
            size=self.group_offset.nbytes
        )

        self.memobj_out_index = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.output_index
        )
        self.prefixes_buf = cl.Buffer(
            self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=prefix_buf
        )
        self.prefix_lengths_buf = cl.Buffer(
            self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=prefix_lengths
        )
        self.suffixes_buf = cl.Buffer(
            self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=suffix_buf
        )
        self.suffix_lengths_buf = cl.Buffer(
            self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=suffix_lengths
        )

        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output)
        self.kernel.set_arg(2, self.memobj_occupied_bytes)
        self.kernel.set_arg(3, self.memobj_group_offset)
        self.kernel.set_arg(4, self.prefixes_buf)
        self.kernel.set_arg(5, self.prefix_lengths_buf)
        self.kernel.set_arg(6, self.suffixes_buf)
        self.kernel.set_arg(7, self.suffix_lengths_buf)
        self.kernel.set_arg(8, pair_count)
        self.kernel.set_arg(9, case_sensitive)
        self.kernel.set_arg(10, self.memobj_out_index)

    def find(self, log_stats: bool = True):
        start_time = time.time()

        self.output_index[0] = 0
        cl.enqueue_copy(self.command_queue, self.memobj_out_index, self.output_index)
        self.command_queue.finish()

        offset = np.array([self.index], dtype=np.uint32)

        cl.enqueue_copy(self.command_queue, self.memobj_group_offset, offset)
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        self.command_queue.finish()

        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        evt = cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
        )
        evt.wait()

        cl.enqueue_copy(self.command_queue, self.output, self.memobj_output).wait()
        cl.enqueue_copy(self.command_queue, self.output_index, self.memobj_out_index).wait()

        self.prev_time = time.time() - start_time

        if log_stats:
            logger.info(f"prefix_suffix_pairs: {self.prefix_suffix_pairs}")
            logger.info(
                f"GPU {self.display_index} Speed: {global_worker_size / ((time.time() - start_time) * 1e6):.2f} MH/s"
            )

        results = []
        if self.output_index[0] == 1:
            length = self.output[0]
            seed = self.output[1: 65].copy()
            results.append((length, seed))

        self.setting.increase_key32()

        return results
