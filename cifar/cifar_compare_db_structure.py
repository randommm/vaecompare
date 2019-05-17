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
    db = SqliteDatabase('results_compare.sqlite3')

class CIFARResult(Model):
    # Data settings
    category1 = IntegerField()
    category2 = IntegerField()

    # Estimation settings
    samples = BlobField()
    elapsed_time = DoubleField()

    class Meta:
        database = db
        indexes = (
            (('category1', 'category2'), True),
        )
CIFARResult.create_table()
