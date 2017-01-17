# Default parameters. You can change any parameter at command line with a
# syntax like
# python cylinder3d.py --Lattice3d=D3Q19
# Parallel run:
# python cylinder3d.py --thread_per_proc=4 --grid3d="[1,2,2]"
import util

numproc = 1
thread_per_proc = 1
myrank = 0
grid2d = [1,1]
grid3d = [1,1,1]

Lattice2d = "D2Q9"
Lattice3d = "D3Q19"

colormap = "hot"
#colormap = "Reds"

remote_output = False
send_host, send_port = "*", 5200
recv_host, recv_port = "jlatt-VirtualBox", 5200

# Read command-line parameters (and list all possible configurable parameters)
util.userparam(["numproc", "thread_per_proc", "grid2d", "grid3d",
                "remote_output", "Lattice2d", "Lattice3d",
                "send_host", "send_port", "recv_host", "recv_port"], locals())

