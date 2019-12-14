-- [X] Query 1 - Yankee Managers
-- List the name of everyone who has managed the New York Yankees.
select
  distinct x.nameFirst,
  x.namelast
from
  managers m
  join teams t on m.teamID = t.teamID
  and m.yearid = t.yearid
  join master x on x.masterid = m.masterid
  and t.name = 'New York Yankees';
-- [X] Query 2 - Yankee Batters
  -- For each year, list how many different players batted for
  -- the New York Yankees.
select
  count(players.masterid),
  players.yearid
from
  (
    select
      distinct p.masterid,
      p.yearid
    from
      teams t
      join batting p on t.teamid = p.teamid
      and t.yearid = p.yearid
    where
      t.name = 'New York Yankees'
  ) as players
group by
  players.yearid
order by
  players.yearid;
-- [X] Query 3 - World Series Losers
  -- List the name of each team that played in but lost the world series
  -- and number of world series that it has lost
  -- (the column WSWin in the Teams table has a N value if the team did not win
  -- the world series in that season, and has a Y in the LgWin column indicating
  -- it won the league). Each winner should be listed just once.
select
  t.name,
  count(t.name) as "losses"
from
  teams t
where
  t.WSWin = 'N'
  and t.LgWin = 'Y'
group by
  t.name
order by
  2 desc;
-- [X] Query 4 - Good Hitters
  -- List the name of each player with more than 200 hits in a season in their
  -- career (hits made is the "H" column in the "Batting" table).
select
  x.nameFirst,
  x.nameLast
from
  master x,
  (
    select
      distinct p.masterid
    from
      batting p
    where
      p.h > 200
  ) as hitters
where
  x.masterid = hitters.masterid;
-- [X] Query 5 - Team Roger
  -- List all the pitchers who have a first name of Roger.
select
  distinct x.nameFirst,
  x.nameLast
from
  master x
  join pitching p on x.masterid = p.masterid
where
  x.nameFirst = 'Roger';
-- [X] Query 6 - Atlanta Braves centerfielders
  -- List the first name and last name of every player that has played
  -- centerfield (use the Appearances table) at any time in their career
  -- for the Atlanta Braves. List each player only once.
select
  distinct x.nameFirst,
  x.nameLast
from
  master x
  join appearances p on x.masterid = p.masterid
  join teams t on t.teamid = p.teamid
where
  p.G_cf > 0
  and t.name = 'Atlanta Braves';
-- [X] Query 7 - Winning streaks
  -- Which teams and years won more games in each year for five consecutive years?
select
  a.name as "Team",
  a.yearid as "Year 1",
  e.yearid as "Year 5",
  a.w as "Wins in year 1",
  b.w as "Wins in year 2",
  c.w as "Wins in year 3",
  d.w as "Wins in year 4",
  e.w as "Wins in year 5"
from
  teams a
  join teams b on a.w > b.w
  and a.yearid = b.yearid + 1
  and a.lgid = b.lgid
  and a.teamid = b.teamid
  join teams c on b.w > c.w
  and b.yearid = c.yearid + 1
  and b.lgid = c.lgid
  and b.teamid = c.teamid
  join teams d on c.w > d.w
  and c.yearid = d.yearid + 1
  and c.lgid = d.lgid
  and c.teamid = d.teamid
  join teams e on d.w > e.w
  and d.yearid = e.yearid + 1
  and d.lgid = e.lgid
  and d.teamid = e.teamid;
-- [X] Query 8 - Fire Sale Teams
  -- List the total salary for two consecutive years, team name, and year for
  -- every team that had a total salary which was less than half that for the
  -- previous year.
  with tsalaries as (
    select
      teamid,
      sum(salary) as salary,
      yearid,
      lgid
    from
      salaries
    group by
      teamid,
      yearid,
      lgid
  )
select
  t.name as Team,
  a.lgid as League,
  a.yearid,
  a.salary,
  b.yearid,
  b.salary as "previous salary"
from
  tsalaries a
  join tsalaries b on a.teamid = b.teamid
  and a.yearid = b.yearid + 1
  and a.salary * 2 < b.salary
  and a.lgid = b.lgid
  join teams t on a.teamid = t.teamid
  and a.yearid = t.yearid
order by
  a.yearid;
-- [X] Query 9 - Returning players
  -- Let a stint be defined as a consecutive number of years in which a player
  -- appeared for some team (not necessarily the same team) and then did not
  -- appear for any team. List all players that had at least five stints.
  with stints as (
    select
      distinct a.masterid,
      a.yearid
    from
      appearances a
    where
      a.yearid + 1 not in (
        select
          distinct yearid
        from
          appearances b
        where
          a.masterid = b.masterid
      )
  ),
  info as (
    select
      s.masterid,
      count(*) as stints
    from
      stints s
      join master m on s.masterid = m.masterid
    group by
      s.masterid
  )
select
  nameFirst,
  NameLast,
  stints
from
  info i
  join master m on i.masterid = m.masterid
  stints >= 5;
-- [X] Query 10 - Brooklyn Dodgers Pitchers
  -- List the first name and last name of every player that has pitched for the
  -- team named the "Brooklyn Dodgers". List each player only once.
select
  distinct x.nameFirst,
  x.namelast
from
  pitching p
  join teams t on p.teamid = t.teamid
  and p.yearid = t.yearid
  join master x on x.masterid = p.masterid
where
  t.name = 'Brooklyn Dodgers';
-- [X] Query 11 - Winningest Teams
  -- List the winning percentage (wins divided by (wins + losses)) over a team's
  -- entire history. Consider a "team" to be a team with the same name, so if the
  -- team changes name, it is considered to be two different teams. Show the team
  -- name and win percentage.
  with calc as (
    select
      cast(sum(t.w) as dec) as TotalWins,
      cast(sum(t.L) as dec) as TotalLosses,
      t.name as TeamName
    from
      teams t
    group by
      t.name
  )
select
  TeamName as "Team Name",
  round(TotalWins / (TotalWins + TotalLosses) * 100, 2) as "Win Percentage"
from
  calc
order by
  1;
-- [X] Query 12 - Triples ranking
  -- Rank players by the number of triples (column 3B in the Batting table) they
  -- have hit in any season and list the top ten such rankings of players.
  -- For instance, the player(s) who hit the most in a season would have rank 1,
  -- the second most would be rank 2, etc.
  -- Note there could be several players in a rank, e.g. five players with rank 4.
  with tripples as(
    select
      masterid,
      threb,
      yearid,
      dense_rank() over (
        order by
          threb desc
      ) as "rank"
    from
      batting b
    where
      threb is not null
  )
select
  nameFirst,
  nameLast,
  "rank",
  yearid
from
  tripples t
  join master m on m.masterid = t.masterid
where
  "rank" <= 10;
-- [X] Query 13 - Yankee Run Kings
  -- List the name, year, and number of home runs hit for each New York Yankee
  -- batter, but only if they hit the most home runs for any player in that season
  with bestHR as (
    select
      yearid,
      max(hr) as hr
    from
      batting
    group by
      yearid
  )
select
  nameFirst,
  nameLast,
  h.hr,
  h.yearid
from
  batting b
  join bestHR h on b.hr = h.hr
  and b.yearid = h.yearid
  join teams t on b.teamid = t.teamid
  and b.yearid = t.yearid
  and b.lgid = t.lgid
  join master m on b.masterid = m.masterid
where
  t.name = 'New York Yankees'
order by
  h.yearid;
-- [X] Query 14 - Brooklyn Dodgers Only
  -- List the first name and last name of every player that has played only for
  -- the Brooklyn Dodgers (i.e., they did not play for any other team including
  -- the Los Angeles Dodgers, note that the Brooklyn Dodgers became the
  -- Los Angeles Dodgers in the 1950s). List each player only once.
select
  distinct nameFirst,
  nameLast
from
  appearances p
  join teams t on p.teamid = t.teamid
  and p.yearid = t.yearid
  and p.lgid = t.lgid
  join master m on m.masterid = p.masterid
where
  t.name = 'Brooklyn Dodgers'
  and p.masterid not in (
    select
      distinct p.masterid
    from
      appearances p
      join teams t on p.teamid = t.teamid
      and p.yearid = t.yearid
      and p.lgid = t.lgid
    where
      t.name != 'Brooklyn Dodgers'
  )
order by
  2;
-- [X] Query 15 - Third best home run hitters each year
  -- List the first name, last name, year and number of home runs
  -- (column HR in the Batting table) of every player that hit the third most
  -- number of home runs for that year. Order by the year.
  with bat as(
    select
      dense_rank() over (
        partition by yearid
        order by
          hr desc
      ) as "rank",
      masterid,
      teamid,
      yearid,
      hr
    from
      batting
    where
      hr is not null
  )
select
  nameFirst,
  nameLast,
  yearid,
  hr
from
  bat b
  join master m on m.masterid = b.masterid
where
  "rank" = 3;
-- [X] Query 16 - Two degrees from Yogi Berra
  -- List the name of each player who appeared on a team with a player that was
  -- at one time was a teamate of Yogi Berra. So suppose player A was a teamate
  -- of Yogi Berra. Then player A is one-degree of separation from Yogi Berra.
  -- Let player B be related to player A because B played on a team in the same
  -- year with player A. Then player B is two-degrees of separation.
  --
  -- teams and years Yogi Berra for
  with yogi as (
    select
      teamid,
      a.yearid,
      a.masterid,
      a.lgid
    from
      appearances a
      join master m on a.masterid = m.masterid
    where
      nameFirst = 'Yogi'
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
      yogi y
      join appearances a on a.yearid = y.yearid
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
      degree1 d
      join appearances a on a.masterid = d.masterid
  ) --
  -- people two degrees from Yogi Berra
select
  distinct nameFirst,
  nameLast
from
  degree1allyears d
  join appearances a on a.teamid = d.teamid
  and a.yearid = d.yearid
  join master m on a.masterid = m.masterid
where
  a.masterid not in (
    select
      masterid
    from
      degree1
  )
order by
  2;
-- [X] Query 17 - Traveling with Rickey
  -- List all of the teams for which Rickey Henderson did not play.
  -- Note that because teams come and go, limit your answer to only the teams
  -- that were in existence while Rickey Henderson was a player.
  -- List each such team once.
  with teams_during_rickey_career as (
    select
      distinct name
    from
      teams
    where
      teams.yearid >= (
        select
          min(yearid)
        from
          appearances a
          join master m on m.masterid = a.masterid
        where
          m.namefirst = 'Rickey'
          and m.namelast = 'Henderson'
        group by
          m.masterid
      )
      and teams.yearid <= (
        select
          max(yearid)
        from
          appearances a
          join master m on m.masterid = a.masterid
        where
          m.namefirst = 'Rickey'
          and m.namelast = 'Henderson'
        group by
          m.masterid
      )
  ),
  teams_rickey_played_for as (
    select
      distinct t.name
    from
      teams t
      join appearances a on a.yearid = t.yearid
      and a.teamid = t.teamid
      and a.lgid = t.lgid
      join master m on a.masterid = m.masterid
    where
      m.namefirst = 'Rickey'
      and m.namelast = 'Henderson'
  )
select
  distinct name
from
  teams_during_rickey_career t
where
  t.name not in (
    select
      *
    from
      teams_rickey_played_for
  )
order by
  1;
-- [X] Query 18 - Median team wins
  -- For the 1970s, list the team name for teams in the National League ("NL")
  -- that had the median number of total wins in the decade (1970-1979 inclusive)
  -- (if there are even number of teams, round up to find the median).
  with total_wins as (
    select
      sum(W) as total_wins,
      name,
      rank() over (
        order by
          sum(W)
      ) as "rank"
    from
      teams t
    where
      t.lgid = 'NL'
      and t.yearid >= 1970
      and t.yearid <= 1979
    group by
      name
  )
select
  name,
  rank
from
  total_wins
where
  rank = (
    select
      count(*)
    from
      total_wins
  ) / 2;
