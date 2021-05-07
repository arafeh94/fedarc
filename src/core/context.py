from abc import ABC, abstractmethod

from mpi4py import MPI

from fedml_api.model.linear.lr import LogisticRegression


class COMM(ABC):
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def pid(self) -> int:
        pass

    @abstractmethod
    def send(self, pid, message, tag=0):
        pass

    @abstractmethod
    def recv(self, src=None, tag=None) -> any:
        pass


class MPI_COMM(COMM):
    def __init__(self):
        self.mpi: MPI.Intracomm = MPI.COMM_WORLD

    def size(self):
        return self.mpi.size

    def pid(self):
        return self.mpi.rank

    def send(self, pid, message, tag=0):
        self.mpi.send(message, pid, tag)

    def recv(self, src=None, tag=None):
        return self.mpi.recv(None, src, tag)


class LOCAL_COMM(COMM):
    SIZE = 10

    def __init__(self):
        self.buffer = []

    def size(self) -> int:
        return LOCAL_COMM.SIZE

    def pid(self) -> int:
        return 0

    def send(self, pid, message, tag=0):
        self.buffer.append(message)

    def recv(self, src=None, tag=None) -> any:
        return self.buffer.pop()


class Context:
    def __init__(self, comm: COMM, **kwargs):
        self.comm = comm

    class Builder:
        def __init__(self, comm: str, **kwargs):
            self.comm: COMM = self.comm_builder(comm)
            self.args = kwargs

        @staticmethod
        def comm_builder(comm: str) -> COMM:
            if comm == 'mpi':
                return MPI_COMM()
            if comm == 'local':
                return LOCAL_COMM()
            return MPI_COMM()

        def args_builder(self, kwargs):
            pass

        def build(self):
            return Context(self.comm, **self.args)
