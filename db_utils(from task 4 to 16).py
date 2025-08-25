# db_utils.py
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd

def fetch_table_as_dataframe(table_name=None, custom_query=None):
    conn = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="root",
        database="abidjan_ai",
        cursorclass=DictCursor
    )
    try:
        c = conn.cursor()
        if custom_query:
            c.execute(custom_query)
        elif table_name:
            c.execute(f'SELECT * FROM {table_name}')
        else:
            raise ValueError("Provide either a table_name or a custom_query.")
        
        data = c.fetchall()
        df = pd.DataFrame(data)
        return df
    finally:
        c.close()
        conn.close()
