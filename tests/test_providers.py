import src.api.pipes as pipes
from src.api.sources import MnistStreamSource
from src.core.aoi.abs_streamer import Streamer

source = MnistStreamSource("localhost", "root", "root", "mnist")
streamer = Streamer(source)
streamer.add_pipe(pipes.MnistDataParser())
streamer.add_pipe(pipes.ToTensor())
streamer.add_pipe(pipes.Batch(5))
results, _ = streamer.next(id='f_00001')
for x, y in results:
    print(x)
    print(y)

