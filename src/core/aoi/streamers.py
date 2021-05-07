from abc import abstractmethod

from src.core.aoi.abs_streamer import StreamSource
import mysql.connector


class FileStreamSource(StreamSource):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.reader = None

    def read(self, **kwargs):
        return self.reader.readline()

    def open(self):
        self.reader = open(self.file_path, "r")

    def close(self):
        self.reader.close()


class MySQLSource(StreamSource):
    def __init__(self, host, user, password, database):
        self.db = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

    def read(self, **kwargs) -> any:
        cursor = self.db.cursor()
        try:
            cursor.execute(self.query(**kwargs))
            result = cursor.fetchall()
            return result
        except Exception:
            pass
        finally:
            cursor.close()
        return None

    @abstractmethod
    def query(self, **kwargs) -> str:
        pass

    def open(self):
        pass

    def close(self):
        self.db.close()
