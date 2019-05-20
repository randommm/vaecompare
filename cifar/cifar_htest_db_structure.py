from peewee import *
import os

try:
    pgdb = os.environ['pgdb']
    pguser = os.environ['pguser']
    pgpass = os.environ['pgpass']
    pghost = os.environ['pghost']
    pgport = os.environ['pgport']

    db = PostgresqlDatabase(pgdb, user=pguser, password=pgpass,
    host=pghost, port=pgport)
except KeyError:
    db = SqliteDatabase('results.sqlite3')

class CIFARHTestResult(Model):
    # Data settings
    category1 = IntegerField()
    category2 = IntegerField()

    # Estimation settings
    pvalue = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db
        indexes = (
            (('category1', 'category2'), True),
        )
CIFARHTestResult.create_table()
