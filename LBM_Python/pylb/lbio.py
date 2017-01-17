import config
import messages
import numpy as np
if not config.remote_output:
    from matplotlib import cm
    import matplotlib.pyplot as plt
import atexit


def is_io_server():
    return True

@atexit.register
def send_end_signal():
    if is_io_server() and config.remote_output:
        remote_send.complete()


if is_io_server() and config.remote_output:
    remote_send = messages.RemoteSend(host=config.send_host, port=config.send_port)


def writelog(*messages):
    if is_io_server():
        if config.remote_output:
            remote_send.send("writelog")
            remote_send.send(messages)
        else:
            print("".join([str(m) for m in messages]))


def save(fname, data):
    if is_io_server():
        if config.remote_output:
            remote_send.send("save")
            remote_send.send(fname)
            remote_send.send(data)
        else:
            np.save(fname, data)


class Plot(object):
    def __init__(self, data, colormap=True):
        self.iniplot(data, colormap)

    def iniplot(self, data, colormap):
        if is_io_server():
            if config.remote_output:
                remote_send.send("iniplot")
                remote_send.send(data)
            else:
                plt.ion()
                plt.hold(False) 
                self.image = plt.imshow(data.transpose(), cmap=getattr(cm, config.colormap), animated=True)
                if colormap:
                    plt.colorbar()

    def draw(self, data):
        if is_io_server():
            if config.remote_output:
                remote_send.send("draw")
                remote_send.send(data)
            else:
                self.image.set_data(data.transpose())
                self.image.autoscale()
                plt.draw()

    def savefig(self, name):
        if is_io_server():
            if config.remote_output:
                remote_send.send("savefig")
                remote_send.send(name)
            else:
                plt.savefig(name)
