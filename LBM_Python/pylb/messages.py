import multiprocessing as mp
import numpy as np
import zmq


class QueueAsPipe:
    def __init__(self):
        self.queue = mp.Queue()

    def send(self, data):
        # Make a copy to guarantee data is not modified before it is sent.
        self.queue.put(data.copy())

    def recv(self, data):
        data[:] = self.queue.get()


class SerialPipe:
    def __init__(self):
        self.buf = np.array([])

    def send(self, data):
        self.buf.resize(data.shape)
        self.buf[:] = data

    def recv(self, data):
        data[:] = self.buf


class ShmemPipe:
    def __init__(self, size):
        if size==0: return
        self.array = mp.Array('d', size, lock=False)
        self.ready_send, self.ready_recv = mp.Event(), mp.Event()
        self.ready_send.set()
        self.buf = np.frombuffer(self.array)

    def send(self, message):
        if len(message)==0: return
        self.ready_send.wait()
        self.ready_send.clear()
        self.buf[:] = message
        self.ready_recv.set()

    def recv(self, message):
        if len(message)==0: return
        self.ready_recv.wait()
        self.ready_recv.clear()
        message[:] = self.buf
        self.ready_send.set()


class ZmqPipe:
    def __init__(self, send_host, recv_host, tag):
        self.send_host = send_host
        self.recv_host = recv_host
        self.tag = tag
        self.sender = None
        self.recver = None

    def send(self, message):
        if self.sender is None:
            self.sender = RemoteSend(self.send_host, self.tag + 5020)
        self.sender.send(message)

    def recv(self, message):
        if self.recver is None:
            self.recver = RemoteRecv(self.recv_host, self.tag + 5020)
        message[:] = self.recver.recv()

    def complete(self):
        if self.sender is not None:
            self.sender.complete()
            self.sender = None


# The ZeroMQ send instance is launched in a separate thread to make
# sends non-blocking. Communication between the main thread and the
# ZeroMQ send instance is achieved through the "recv_conn" pipe.
def zmq_sender(host, port, recv_conn):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://{host}:{port}".format(host=host, port=port))
    message = ""
    while not message == "end":
        message = recv_conn.get()

        #message = recv_conn.recv()
        if isinstance(message, np.ndarray):
            meta = {"dtype": str(message.dtype), "shape": message.shape}
            socket.send_json(meta)
            socket.send(message)
        else:
            socket.send_json(message)


class RemoteSend(object):
    def __init__(self, host, port):
        q = mp.Queue()
        self.send_conn = q
        recv_conn = q
        self.p = mp.Process(target=zmq_sender, args=(host, port, recv_conn))
        self.p.start()

    def send(self, message):
        self.send_conn.put(message)

    def complete(self):
        self.send("end")
        self.p.join()


class RemoteRecv(object):
    def __init__(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{host}:{port}".format(host=host, port=port))
        self.port = port

    def recv(self):
	message = self.socket.recv_json()
        try:
           dtype, shape = message['dtype'], message['shape']
           raw_array = self.socket.recv()
           return np.frombuffer(buffer(raw_array), dtype=dtype).reshape(shape)
        except TypeError:
           return message
