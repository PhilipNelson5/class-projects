#!/usr/bin/env bash

#sqlformat -r -k lower -i lower queries.sql > tmp

/usr/bin/sql-formatter-cli < queries.sql > tmp
mv tmp queries.sql
