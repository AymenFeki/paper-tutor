"""Database connection shared by all scripts."""

import os

import psycopg
from dotenv import load_dotenv


def connect():
    load_dotenv()
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="paper_tutor",
        user="tutor",
        password=os.getenv("POSTGRES_PASSWORD"),
    )