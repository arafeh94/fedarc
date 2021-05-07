from mpi4py import MPI

from src.core.aoi.abs_streamer import Streamer, StreamSource
import mysql.connector

from src.core.aoi.streamers import MySQLSource
from src.core.context import Context


class MnistStreamSource(MySQLSource):
    def query(self, **kwargs) -> str:
        if 'id' not in kwargs:
            raise Exception('mnist source needs {id} parameter')
        client_id = kwargs['id']
        return "select * from skewed where user_id=" + str(client_id)


class ModelReceiver(StreamSource):
    def read(self, **kwargs):
        context: Context = kwargs.get('context', None)
        if context is None:
            raise Exception('context is not defined')
        return context.comm.recv()


class FakeRandomModelReceiver(StreamSource):
    def read(self, **kwargs) -> any:
        pass
