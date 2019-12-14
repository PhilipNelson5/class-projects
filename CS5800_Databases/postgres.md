Make something accessible to postgres user
```
sudo mkdir /path/to/dir
sudo chmod 775 /path/to/dir
sudo chown postgres /path/to/dir
```

Get on to the postgres user
`sudo su - postgres`

Start the postgres database server
`postgres -D /usr/local/pgsql/data`
in the background with a log
`postgres -D /usr/local/pgsql/data >logfile 2>&1 &`
from your own user
`su postgres -c 'pg_ctl start -D /usr/local/pgsql/data -l serverlog'`
