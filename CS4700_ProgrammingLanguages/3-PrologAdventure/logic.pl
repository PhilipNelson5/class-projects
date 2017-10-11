:-include(adventure).
reset   :- write("\033[0m").
black   :- write("\033[30m").
red     :- write("\033[31m").
green   :- write("\033[32m").
yellow  :- write("\033[33m").
blue    :- write("\033[34m").
magenta :- write("\033[35m").
cyan    :- write("\033[36m").
white   :- write("\033[37m").
bold    :- write("\033[1m").
clear   :- write("\033[2J").

connected(X,Y):- door(X,Y).
connected(X,Y):- door(Y,X).

% HW3 --------------------------------------------------------------------------------------------------------------------------------



% HW2 --------------------------------------------------------------------------------------------------------------------------------

look:- here(X), look(X), !.
checkLook(Place):- here(Location), existsHere(Place, Location), look(Place).
checkLook(Object):- has(Object), look(Object).

checkStudy(Object):- here(Location), existsHere(Object, Location), study(Object), write("1"), !.
checkStudy(Thing):- has(Thing), study(Thing), write("2"), !.

checkInventory:- inventory.
inventory:- blue, write("Inventory:"), nl, reset, has(Item), printName(Item), nl, fail.
inventory:- true.

move(Place):- here(Cur), retract(here(Cur)), asserta(here(Place)), look(Place), !.
checkMove(Place):- here(Cur), connected(Place, Cur), puzzle(Place), move(Place), !.

take(Item):- location(Item,Loc), retract(location(Item, Loc)), asserta(has(Item)), inventory.
checkTake(Item):- here(Cur), isHere(Item, Cur), not(heavy(Item)), take(Item).

isHere(Item, Place):- location(Item, Place), !.
isHere(Item, Place):- location(Item, Container), isHere(Container, Place).

put(Item, Loc):- retract(has(Item)), asserta(location(Item, Loc)).
checkPut(Item, Loc):- (room(Loc); container(Loc)), here(Cur), existsHere(Loc, Cur), put(Item, Loc).

existsHere(Place1, Place2):- Place1 == Place2, !.
existsHere(Place1, Place2):- location(Place1, Place2), !.
existsHere(Place1, Place2):- location(Place1, Place3), existsHere(Place3, Place2).

% HW1 --------------------------------------------------------------------------------------------------------------------------------

look(Place):- room(Place), yellow, write("Location:\n"), reset, descriptionShort(Place), listConnecions(Place), listItems(Place), !.
look(Place):- location(Place, _), descriptionShort(Place), !.
look(_).

study(Object):- container(Object), yellow, write("Container:\n"), reset, descriptionLong(Object), nl, magenta, write("\nContains:\n"), reset, listContainter(Object), !.
study(Object):- descriptionLong(Object), nl, fail.
study(_).

/* List contents of a container */
listContainter(Container):- location(Item, Container), descriptionShort(Item), nl, fail.
listContainter(_).

/* List connections from a room */
listConnecions(Place):- room(Place), cyan, write("\n\nConnections:\n"), reset, connected(Place,X), descriptionShort(X), nl, fail.
listConnecions(_).

/* Display the long description of a room */
descriptionLong(Place):- printName(Place), write(": "), long_desc(Place, Descript), write(Descript), !.
descriptionLong(_).

/* Display the short description of a room */
descriptionShort(Place):- printName(Place), write(": "), short_desc(Place, Descript), write(Descript), !.
descriptionShort(_).

/* List the items in a room */
listItems(Place):- room(Place), red, write("\nItems:\n"), reset, location(Item, Place), descriptionShort(Item), nl, fail.
listItems(_).

printName(Thing):- name(Thing, Name), white, write(Name), reset, !.
