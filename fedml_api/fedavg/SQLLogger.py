import json

import mysql.connector

try:
    _mysql_logger = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="logging"
    )
except mysql.connector.Error as err:
    _mysql_logger = None
    print("logging database isn't created")
    print("create it by executing this sql")
    print(
        "create table logs(	id int auto_increment		"
        "primary key,	type varchar(30) null,	"
        "content text null,	created_date timestamp "
        "default CURRENT_TIMESTAMP not null);"
    )


def _cache_log(type: str, content: str):
    if _mysql_logger is None:
        return False
    query = "insert into logs (type, content) value (%s,%s)"
    value = (type, content)
    _mysql_logger.cursor().execute(query, value)
    _mysql_logger.commit()
    return True


def _log_json(obj):
    _cache_log("json", json.dumps(obj))


def create_cache():
    return LogCache()


class LogCache:
    def __init__(self):
        self.cache = {}

    def put(self, key, value):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def save(self):
        _log_json(self.cache)
        self.clear()
