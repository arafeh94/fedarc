import threading
from abc import abstractmethod
from copy import copy


class StreamSource:
    @abstractmethod
    def read(self, **kwargs) -> any:
        pass

    def open(self):
        pass

    def close(self):
        pass


class Pipe:
    def __init__(self):
        self.source = None

    @abstractmethod
    def next(self, data, raw, **kwargs) -> object:
        pass

    def on_close(self):
        pass


class Streamer:
    STATUS_STOPPED = '1'
    STATUS_RUNNING = '0'
    STATUS_CLOSED = '2'

    def __init__(self, source: StreamSource, **kwargs):
        self.source = source
        self.pipes = []
        self.status = Streamer.STATUS_STOPPED
        self.thread = None
        self.buffer = []
        self.log = kwargs.get('log', False)

    def add_pipe(self, pipe: Pipe):
        pipe.source = self
        self.pipes.append(pipe)
        return self

    def _next_reading(self, **kwargs):
        if len(self.buffer) > 0:
            return self.buffer.pop(0)
        return self.source.read(**kwargs)

    def next(self, **kwargs) -> (any, any):
        reading = self._next_reading(**kwargs)
        pipe_result = copy(reading)
        if reading is not None:
            pipe_result = reading
            for pipe in self.pipes:
                pipe_result = pipe.next(pipe_result, reading, **kwargs)
                if pipe_result is None:
                    break
        return pipe_result, reading

    def run(self, init: callable = None, **kwargs):
        if Streamer.STATUS_RUNNING:
            return
        self.status = Streamer.STATUS_RUNNING
        self.source.open()
        init(**kwargs)
        while self.status == Streamer.STATUS_RUNNING:
            self.next(**kwargs)

    def stop(self):
        self.status = Streamer.STATUS_STOPPED
        for pipe in self.pipes:
            pipe.on_close()
        self.source.close()
