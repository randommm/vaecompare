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
    db = SqliteDatabase('results_htest.sqlite3')

class Result(Model):
    # Data settings
    distribution = IntegerField()
    no_instances = IntegerField()
    dissimilarity = DoubleField()
    random_seed = IntegerField()

    # Estimation settings
    pvalue = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db
        indexes = (
            (('distribution', 'no_instances', 'dissimilarity',
              'random_seed'), True),
        )
Result.create_table()
