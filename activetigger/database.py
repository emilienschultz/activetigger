# Database management
import sqlite3
from pathlib import Path

class DataBase():
    def __init__(self,db_name):
        file = db_name+".sqlite"
        if Path(file).exists():
            self.db = sqlite3.connect(file)
        else:
            self.db = self.create_database(file)

    def create_database(self, db_name):
        con = sqlite3.connect(db_name)
        
        con.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT,
                params TEXT
            )
            ''')
        
        con.execute('''
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY,
                project TEXT,
                text TEXT,
                label TEXT,
                user INT,
                emb_sbert TEXT,
                emb_fasttext TEXT
            )
            ''')
        
        con.execute('''
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY,
                text TEXT,
                label TEXT
            )
            ''')
        
        # pour le futur

        con.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                mail TEXT,
                comment TEXT
            )
            ''')
        
        con.execute('''
            CREATE TABLE IF NOT EXISTS access (
                id INTEGER PRIMARY KEY,
                user TEXT,
                project TEXT
            )
            ''')
        
        con.commit()


        return con