#!/usr/bin/env bash

file=constraints.sql

sqlformat -r -s --wrap_after 80 -k upper -i lower ${file} > tmp

#/usr/bin/sql-formatter-cli < constraints.sql > tmp

mv tmp ${file}
