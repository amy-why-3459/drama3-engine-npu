"""Exercise experimental scheduler shutdown with two real Gloo ranks, no model."""
import datetime
import queue
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, rendezvous: str) -> None:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    dist.init_process_group(
        'gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.tp_size = 2
    scheduler.is_entry_rank = rank == 0
    scheduler.tp_group = SimpleNamespace(rank=rank, ranks=[0, 1])
    scheduler.tp_cpu_group = dist.group.WORLD
    scheduler.inbox = queue.Queue()
    scheduler.idle_wait_message = None
    scheduler.running = True
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.shutdown_resources = Mock()

    # A follower can receive the control-plane stop before the leader does.
    # It must stay in the next broadcast instead of tearing down the group.
    if rank == 1:
        scheduler.stop()
        assert scheduler.running
    else:
        pass
    dist.barrier()
    if rank == 0:
        scheduler.stop()
        assert scheduler.running
    else:
        pass
    scheduler.shutdown_resources.assert_not_called()
    assert scheduler.recv_scheduler_messages() == []
    assert not scheduler.running
    dist.barrier()
    dist.destroy_process_group()
    print(f'RANK_{rank}_COLLECTIVE_SHUTDOWN_PASS', flush=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='drama3-tp-shutdown-') as directory:
        mp.spawn(worker, args=(f'{directory}/gloo',), nprocs=2, join=True)
    print('TP_SHUTDOWN_PASS', flush=True)
