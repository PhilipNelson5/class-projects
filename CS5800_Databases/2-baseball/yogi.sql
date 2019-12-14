--------------------------------------------------
-- 4873 rows - players two degrees from Yogi Berra
-- including players one degree from Yogi Berra
-- withoug taking into account the first year they played together
--------------------------------------------------
-- teams and years Yogi Berra for
with yogi as (
  select
    teamid,
    a.yearid,
    a.masterid
  from
    appearances a,
    master m
  where
    a.masterid = m.masterid
    and nameFirst = 'Yogi'
    and nameLast = 'Berra'
),
-- people one degree from Yogi Berra
-- and first year they played together
degree1 as (
  select
    a.masterid,
    min(a.yearid) as yearid
  from
    yogi y,
    appearances a
  where
    a.yearid = y.yearid
    and a.teamid = y.teamid
  group by
    a.masterid
),
-- all years of one degree players after they played with Yogi
degree1allyears as (
  select
    a.masterid,
    a.yearid,
    a.teamid,
    d.yearid as firstyear
  from
    degree1 d,
    appearances a
  where
    a.masterid = d.masterid
) --
-- people two degrees from Yogi Berra
select
  distinct nameFirst,
  nameLast
from
  degree1allyears d,
  appearances a,
  master m
where
  a.teamid = d.teamid
  and a.yearid = d.yearid
  and a.masterid not in (
    select
      masterid
    from
      master
    where
      namefirst = 'Yogi'
      and nameLast = 'Berra'
  )
  and a.masterid = m.masterid
order by
  2;
--------------------------------------------------
  -- 4615 rows - players two degrees from Yogi Berra
  -- without players one degree from Yogi Berra
  --------------------------------------------------
  -- teams and years Yogi Berra for
  with yogi as (
    select
      teamid,
      a.yearid,
      a.masterid,
      a.lgid
    from
      appearances a,
      master m
    where
      a.masterid = m.masterid
      and nameFirst = 'Yogi'
      and nameLast = 'Berra'
  ),
  -- people one degree from Yogi Berra
  -- and first year they played together
  degree1 as (
    select
      a.masterid,
      min(a.yearid) as yearid,
      a.lgid
    from
      yogi y,
      appearances a
    where
      a.yearid = y.yearid
      and a.teamid = y.teamid
      and a.lgid = y.lgid
    group by
      a.masterid,
      a.lgid
  ),
  -- all years of one degree players after they played with Yogi
  degree1allyears as (
    select
      a.masterid,
      a.yearid,
      a.teamid,
      d.yearid as firstyear
    from
      degree1 d,
      appearances a
    where
      a.masterid = d.masterid
  ) --
  -- people two degrees from Yogi Berra
select
  distinct nameFirst,
  nameLast
from
  degree1allyears d,
  appearances a,
  master m
where
  a.teamid = d.teamid
  and a.yearid = d.yearid
  and a.masterid not in (
    select
      masterid
    from
      degree1allyears
  )
  and a.masterid = m.masterid
order by
  2;
--------------------------------------------------
  -- 3354 rows - players two degrees from Yogi Berra
  -- without players one degree from Yogi Berra
  -- taking into account the first year they played together
  --------------------------------------------------
  -- teams and years Yogi Berra for
  with yogi as (
    select
      teamid,
      a.yearid,
      a.masterid,
      a.lgid
    from
      appearances a,
      master m
    where
      a.masterid = m.masterid
      and nameFirst = 'Yogi'
      and nameLast = 'Berra'
  ),
  -- people one degree from Yogi Berra
  -- and first year they played together
  degree1 as (
    select
      a.masterid,
      min(a.yearid) as yearid,
      a.lgid
    from
      yogi y,
      appearances a
    where
      a.yearid = y.yearid
      and a.teamid = y.teamid
      and a.lgid = y.lgid
    group by
      a.masterid,
      a.lgid
  ),
  -- all years of one degree players after they played with Yogi
  degree1allyears as (
    select
      a.masterid,
      a.yearid,
      a.teamid,
      d.yearid as firstyear
    from
      degree1 d,
      appearances a
    where
      a.masterid = d.masterid
      and a.yearid >= d.yearid -- Only after they played with Yogi Berra
  ) --
  -- people two degrees from Yogi Berra
select
  distinct nameFirst,
  nameLast
from
  degree1allyears d,
  appearances a,
  master m
where
  a.teamid = d.teamid
  and a.yearid = d.yearid
  and a.masterid not in (
    select
      masterid
    from
      degree1allyears
  )
  and a.masterid = m.masterid
order by
  2;
------------------------------------------------------------------------
------------------------------------------------------------------------
--  ATTEMPT TO CREATE DEGREE ONE AND TWO WITH TEAM AND YEAR INFO
------------------------------------------------------------------------
------------------------------------------------------------------------
-- teams and years Yogi Berra for
with yogi as (
  select
    teamid,
    a.yearid,
    a.masterid
  from
    appearances a,
    master m
  where
    a.masterid = m.masterid
    and nameFirst = 'Yogi'
    and nameLast = 'Berra'
),
-- people one degree from Yogi Berra
-- and first year they played together
degree1 as (
  select
	nameFirst,
	nameLast,
    min(a.yearid) as "first year with yogi",
    a.masterid
  from
    yogi y,
    appearances a,
	master m
  where
    a.yearid = y.yearid
    and a.teamid = y.teamid
	and m.masterid = a.masterid
  group by
    a.masterid,
	m.nameFirst,
	m.nameLast
),
-- all years of one degree players after they played with Yogi
degree1allyears as (
  select
	d.nameFirst,
	d.nameLast,
    a.masterid,
    a.yearid,
    a.teamid,
    d."first year with yogi"
  from
    degree1 d,
    appearances a
  where
    a.masterid = d.masterid
),
-- people two degrees from Yogi Berra
degree2 as (select distinct
  m.nameFirst,
  m.nameLast,
  m.masterid,
  d.teamid,
  a.yearid
  --max(a.yearid) as "last year played with degree one"
from
  degree1allyears d,
  appearances a,
  master m
where
  a.teamid = d.teamid
  and a.yearid = d.yearid
  and a.masterid != (
    select
      masterid
    from
      master
    where
      namefirst = 'Yogi'
      and nameLast = 'Berra'
  )
  and a.masterid = m.masterid
  order by masterid
),
-- players degree one with team information
degree1withteams as (select
  d.masterid,
  d.nameFirst,
  d.nameLast,
  d."first year with yogi",
  t.name,
  t.teamid
from
  degree1 d,
  appearances a,
  teams t
where
  d.masterid = a.masterid
  and d."first year with yogi" = a.yearid
  and d."first year with yogi" = t.yearid
  and a.teamid = t.teamid
  and a.teamid in (select teamid from yogi)
),
-- players degree two with team information
degree2withteams as (select
  d.masterid,
  d.nameFirst,
  d.nameLast,
  d.yearid,
  t.name,
  t.teamid
from
  degree2 d,
  teams t
where
  d.yearid = t.yearid
  and t.teamid = d.teamid
  and t.yearid = d.yearid
),
degree2lastyear as (select
  masterid,
  namefirst,
  namelast,
  max(yearid)
from
  degree2withteams d
group by
  masterid,
  namefirst,
  namelast
order by
  masterid
)

select
  t.*
from
--   degree1withteams t;
--   degree2withteams t;
--   degree2 t;
degree2lastyear t;;
