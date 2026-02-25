import sqlite3
import os
os.makedirs('test_outputs', exist_ok=True)
conn = sqlite3.connect('test_outputs/test.db')
conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, name TEXT)')
conn.execute("INSERT INTO t (name) VALUES (?)", ('hello',))
conn.commit()
print(conn.execute('SELECT * FROM t').fetchall())
conn.close()