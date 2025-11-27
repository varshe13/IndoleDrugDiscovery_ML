import os
import torch
# DDP
from torch.nn.parallel import DistributedDataParallel  as DDP
from torch.distributed import init_process_group, destroy_process_group


# DDP Util function: return a free port for DDP
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


# DDP 
def ddp_setup(backend="nccl"):
    init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def torch_print_all(x):
    """ print the whole tensor in any shape.
        Only print, not log.
    """
    torch.set_printoptions(profile="full")
    print(x) # prints the whole tensor
    torch.set_printoptions(profile="default") # reset
