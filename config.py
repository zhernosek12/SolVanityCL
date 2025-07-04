import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('PG_HOST'),
    'port': os.getenv('PG_PORT'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASS'),
    'dbname': os.getenv('PG_DBNAME')
}

# максимальное количество записей за 1 запрох в очереди GPU
SUBCHUNK_MAX = 100