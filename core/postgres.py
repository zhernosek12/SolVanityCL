import psycopg2

import select

"""

## 1. Create a function to notify!

CREATE OR REPLACE FUNCTION notify_batch_insert() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('new_data_channel', 'batch_inserted');
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

## 2. Create a trigger to intro in database!

CREATE TRIGGER batch_notify_trigger
AFTER INSERT ON wallets_to_work
FOR STATEMENT
EXECUTE FUNCTION notify_batch_insert();

"""


class Postgres:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def start_listen(self, callback):
        with self.conn.cursor() as cur:
            cur.execute("LISTEN new_data_channel;")

            while True:
                if select.select([self.conn], [], [], 5) == ([], [], []):
                    continue
                self.conn.poll()
                while self.conn.notifies:
                    self.conn.notifies.pop(0)

                    with self.conn.cursor() as select_cur:
                        select_cur.execute("""
                            SELECT id, token_address, start_address
                            FROM public.wallets_to_work
                            WHERE status = 'pending'
                        """)
                        rows = select_cur.fetchall()

                    callback(rows)

    def update(self, start_address, private_address, status, row_id):
        self.cursor.execute(f"""
            UPDATE wallets_to_work
            SET 
                start_address = '{start_address}',
                private_address = '{private_address}',
                status = '{status}'
            WHERE id = '{row_id}';
        """)

    def close(self):
        self.cursor.close()
        self.conn.close()
