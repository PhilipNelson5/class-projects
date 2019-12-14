--------------------------------------------------------------------------------
-- title: Constraints in Postgres
-- author: Philip Nelson and Scott Glaittli
-- date: 19th November 2019
--------------------------------------------------------------------------------
-- [X] Constraint 1
-- The default number of hits for a batter is 0.
ALTER TABLE batting
ALTER h SET DEFAULT 0;

-- To test that the constraint works, I inserted the following tuple
-- without a number of hits. After querying that tuple, we can see that
-- "h" has a value of 0.
insert into batting (masterid, yearid)
values ('bondsba01', 2020);

select masterid, h
from batting
where
  masterid = 'bondsba01'
  and  yearid = 2020;

--------------------------------------------------------------------------------
-- [X] Constraint 2
-- A batter cannot have more than HR (home runs) than H (hits).
CREATE OR REPLACE FUNCTION h_and_hr_check()
  RETURNS trigger AS
$BODY$
BEGIN
  IF (select sum(hr)
      from batting b
      where b.masterid = new.masterid) + new.hr
      >
      (select sum(h)
       from batting b
       where b.masterid = new.masterid) + new.h
  THEN
    raise exception 'player can not have more home runs than hits';
  END IF;
  
  RETURN new;
END;
$BODY$
LANGUAGE plpgsql;

CREATE TRIGGER h_and_hr_check_trigger
  BEFORE INSERT OR UPDATE
  ON batting
  FOR EACH ROW
  EXECUTE PROCEDURE h_and_hr_check();

-- To test that the trigger works, I inserted the following tuples for the
-- player with masterid 'aardsda01' who previously had 0 hr and 0 h.
-- The first tuple was inserted successfully but on execution of the second
-- query postgres raised the following error
-- ERROR:  player can not have more home runs than hits
insert into batting (masterid, hr, h)
values ('aardsda01', 10, 20);

insert into batting (masterid, hr, h)
values ('aardsda01', 20, 0);

-- The following query lists all masterid, total hr, total h for batting
with
hrs as (
  select
    masterid, sum(hr) as "total hr"
  from batting
  group by masterid
),
  hs as (
  select
    masterid, sum(h) as "total h"
  from batting
  group by masterid
)
select * 
  from hrs natural join hs
  where
    "total h" is not null
    and "total hr" is not null
  order by masterid;

--------------------------------------------------------------------------------
-- [X] Constraint 3
-- The masterid of a batter must exist in the master table.
ALTER TABLE batting
ADD CONSTRAINT exists_masterid
FOREIGN KEY (masterid) REFERENCES master (masterid);

-- To test that the constraint works I inserted the following tuple.
-- When executing this query, postgres raised this error
-- ERROR:  insert or update on table "batting" violates foreign key
--         constraint "exists_masterid"
-- DETAIL:  Key (masterid)=(foobar) is not present in table "master".
insert into batting (masterid)
values ('foobar');

--------------------------------------------------------------------------------
-- [X] Constraint 4
-- When a team loses more than 161 games in a season, the fans want to forget
-- about the team forever, so all pitching records for the team for that year
-- should be deleted.
create or replace function forget_if_161_losses()
  returns trigger as
$BODY$
BEGIN
  IF (select sum(L) 
      from pitching
      where yearid = new.yearid
      and teamid = new.teamid) > 161
  THEN
    delete from pitching
    where yearid = new.yearid
    and teamid = new.teamid;
  END IF;
  return new;
END;
$BODY$
LANGUAGE plpgsql;

CREATE TRIGGER has_161_losses
  AFTER INSERT OR UPDATE
  ON pitching
  FOR EACH ROW
  EXECUTE PROCEDURE forget_if_161_losses();

-- To test that the trigger works I inserted the following tuple.
insert into pitching(masterid, yearid, teamid, lgid, l)
values('foo', 2020, 'NYA', 'AL', 100)

insert into pitching(masterid, yearid, teamid, lgid, l)
values('bar', 2020, 'NYA', 'AL', 100)

-- then I used the following query after each insertion and verified that after
-- inserting the second tuple, all the records from 2020 for NYA were deleted

select *
from pitching p
where p.yearid = 2020
and p.teamid = 'NYA'
--------------------------------------------------------------------------------
-- [X] Constraint 5
-- If a batter hits 100 HRs in a year, they are automatically inducted into the
-- Hall of Fame (a new row in the hallOfFame table).
create or replace function induct_hall_of_fame_100()
  returns trigger as
$BODY$
BEGIN
  IF (select sum(hr) 
      from batting
      where masterid = new.masterid
      and yearid = new.yearid) >= 100
      
      and not exists (select * from halloffame
                where masterid = new.masterid)
  THEN
    INSERT INTO halloffame (masterid, yearid, needed_note)
    VALUES
       (
          new.masterid,
          new.yearid,
          '100+ hrs in one season'
       );
  END IF;
  return new;
END;
$BODY$
LANGUAGE plpgsql;

CREATE TRIGGER auto_hall_of_fame
  AFTER INSERT OR UPDATE
  ON batting
  FOR EACH ROW
  EXECUTE PROCEDURE induct_hall_of_fame_100();

-- To test that the trigger works I inserted the following tuples.
insert into batting(masterid, yearid, hr, h)
values('jacobar01', 2020, 150, 200)

insert into batting(masterid, yearid, hr, h)
values('jacobar01', 2021, 150, 200)

-- when I ran the following query, the result was
-- jacobar01, 2020, '100+ hrs in one season'
select masterid, yearid, needed_note
from halloffame
where masterid = 'jacobar01'

--------------------------------------------------------------------------------
-- [X] Constraint 6
-- All players must have some nameFirst, i.e., it cannot be null.

-- In order to add this constraint, I needed to remove all the null values
-- that already existed for any row. I set them to the empty string
UPDATE master
SET nameFirst = ''
WHERE nameFirst is null

-- Then I was able to alter the table.
ALTER TABLE master
ALTER COLUMN nameFirst
SET NOT NULL;

-- To test that the constraint works I inserted the following tuple.
-- When executing this query, postgres raised this error
-- ERROR:  null value in column "namefirst" violates not-null constraint
insert into master (masterid, nameLast)
values ('foobar', 'bar')

--------------------------------------------------------------------------------
-- [X] Constraint 7
-- Everybody has a unique name (combined first and last names).

CREATE OR REPLACE FUNCTION is_name_unique() 
  RETURNS trigger as 
$BODY$
BEGIN
  IF exists (select *
             from master
             where nameFirst = new.nameFirst
             and nameLast = new.nameLast)
  THEN
    raise exception 'player with that name already exists';
  END IF;
  
  RETURN new;
END;
$BODY$
LANGUAGE plpgsql;

CREATE TRIGGER unique_name
  BEFORE INSERT OR UPDATE
  ON master
  FOR EACH ROW
  EXECUTE PROCEDURE is_name_unique();

-- To test that the trigger works I inserted the following tuple.
-- When executing this query, postgres raised this error
-- ERROR:  player with that name already exists
insert into master(nameFirst, nameLast)
values('Yogi', 'Berra')
