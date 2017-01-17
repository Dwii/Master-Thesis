# The client is started on the computer which has access to the
# screen. It simply forwards the output from the simulation code.
# Example: on a Desktop computer, type
#   python client.py --recv_host=comp_node --recv_port=5200
# On the computational device "comp_node", type
#   python simulation2d.py --send_host="*" --send_port=5200 --remote_output=True

import lbio
import messages
import config
import inspect

remote_recv = messages.RemoteRecv(host=config.recv_host, port=config.recv_port)

class LocalIO(object):
    def iniplot(self, data):
        self.plot = lbio.Plot(data)

    def draw(self, data):
        self.plot.draw(data)

    def savefig(self, data):
        self.plot.savefig(data)

    def writelog(self, messages):
        print("".join([str(m) for m in messages]))

    def save(self, fname, data):
        lbio.save(fname, data)


local_io = LocalIO()
while True:
    message = remote_recv.recv()
    if message == "end":
        break
    else:
        fun = getattr(local_io, message)
        numargs = len(inspect.getargspec(fun).args)-1
        args = (remote_recv.recv() for i in range(numargs))
        fun(*args)

