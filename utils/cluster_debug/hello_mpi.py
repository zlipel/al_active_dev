from mpi4py import MPI
import socket, os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()
procid = os.getenv("SLURM_PROCID")

print(f"Hello from rank {rank}/{size} on {host}, SLURM_PROCID={procid}")

total = comm.allreduce(rank, op=MPI.SUM)
if rank == 0:
    print(f"[rank 0] allreduce sum = {total}")
