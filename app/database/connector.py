from app.config import Config
import pg8000
from google.cloud.sql.connector import Connector

def get_db_connection() -> pg8000.dbapi.Connection:
    connector = Connector()
    return connector.connect(
        Config.DB_INSTANCE_NAME,
        "pg8000",
        user=Config.DB_USER,
        password=Config.DB_PASS,
        db=Config.DB_NAME,
    )