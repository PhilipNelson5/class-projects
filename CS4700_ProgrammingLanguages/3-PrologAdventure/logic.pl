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

% HW3 -------------------------------------------------------------------------------------------------------------------------

give(Item):- asserta(has(Item)).

isIn(Room):- here(Here), Here == Room.

transfer(Disk, Pylon2):- location(Disk, Pylon1), retract(location(Disk, Pylon1)), asserta(location(Disk, Pylon2)), printTowers, win.
checkTransfer(Disk, Pylon_f):- isIn(secret_lab), location(Pylon_f, _), Disk == small_disk, transfer(Disk, Pylon_f), !.
checkTransfer(Disk, Pylon_f):- isIn(secret_lab), location(Pylon_f, _), Disk == medium_disk, location(medium_disk, Pylon_m), not(location(small_disk, Pylon_m)), not(location(small_disk, Pylon_f)), transfer(Disk, Pylon_f), !.
checkTransfer(Disk, Pylon_f):- isIn(secret_lab), location(Pylon_f, _), Disk == large_disk, location(large_disk, Pylon_l), not(location(small_disk, Pylon_l)), not(location(medium_disk, Pylon_l)), not(location(small_disk, Pylon_f)), not(location(medium_disk, Pylon_f)), transfer(Disk, Pylon_f), !.

hanoi:- red, write("\nred pylon: "), reset, printPylon(pylon_a), nl, blue, write("\nblue pylon: "), reset, printPylon(pylon_b), nl, green, write("\ngreen pylon: "), reset, printPylon(pylon_c), reset, nl,nl, !.
printPylon(Pylon):- location(Disk, Pylon), write(" < "), printNameNC(Disk), write(" > "), fail.
printPylon(_).

make(Product):- create_recipe(_, List, Product), use(List), asserta(has(Product)).
checkMake(Product):- create_recipe(Device, List, Product), here(Here), location(Device, Here), hasAll(List), make(product).

hasAll([]).
hasAll([H|T]):- has(H), hasAll(T).

use([]).
use([H|T]):- has(H), retract(has(H)), use(T).

win:-  location(large_disk, pylon_c), location(medium_disk, pylon_c), location(small_disk, pylon_c), nl, write("You foiled the evil Dr. Sundberg!"),nl, nl.

printTowers:- hanoiTop, nl, hanoiMid, nl, hanoiBot, nl, hanoiLables.

hanoiTop:- red, pylonHas3(pylon_a), blue, pylonHas3(pylon_b), green, pylonHas3(pylon_c).
pylonHas3(Pylon):- location(small_disk, Pylon), location(medium_disk, Pylon), location(large_disk, Pylon), write("   (_)   "), !.
pylonHas3(_):- write("    |    ").

hanoiMid:- red, pylonHas2(pylon_a), blue, pylonHas2(pylon_b), green, pylonHas2(pylon_c).
pylonHas2(Pylon):- location(small_disk, Pylon), location(medium_disk, Pylon), not(location(large_disk, Pylon)), write("   (_)   "), !.
pylonHas2(Pylon):- location(medium_disk, Pylon), location(large_disk, Pylon), write("  (___)  "), !.
pylonHas2(_):- write("    |    ").

hanoiBot:- red, pylonHas1(pylon_a), blue, pylonHas1(pylon_b), green, pylonHas1(pylon_c).
pylonHas1(Pylon):- location(small_disk, Pylon), not(location(medium_disk, Pylon)), not(location(large_disk, Pylon)), write("   (_)   "), !.
pylonHas1(Pylon):- location(medium_disk, Pylon), not(location(large_disk, Pylon)), write("  (___)  "), !.
pylonHas1(Pylon):- location(large_disk, Pylon), write(" (_____) "), !.
pylonHas1(_):- write("    |    ").

hanoiLables:- red, write("¯¯¯¯¯¯¯¯¯"), blue, write("¯¯¯¯¯¯¯¯¯"), green, write("¯¯¯¯¯¯¯¯¯").

% HW2 -------------------------------------------------------------------------------------------------------------------------

look:- here(X), look(X), !.

checkLook(Place):- here(Here), existsHere(Place, Here), look(Place), !.
checkLook(Object):- has(Object), look(Object), !.

checkStudy(Object):- here(Here), existsHere(Object, Here), study(Object), !.
checkStudy(Thing):- has(Thing), study(Thing), !.

checkInventory:- inventory.
inventory:- blue, write("Inventory:"), nl, reset, has(Item), printName(Item), nl, fail.
inventory:- true.

move(Place):- here(Here), retract(here(Here)), asserta(here(Place)), look(Place), !.
checkMove(Place):- here(Here), connected(Place, Here), puzzle(Place), move(Place), !.

take(Item):- location(Item,Loc), retract(location(Item, Loc)), asserta(has(Item)), inventory.
checkTake(Item):- here(Here), isHere(Item, Here), not(heavy(Item)), take(Item).

isHere(Item, Place):- location(Item, Place), !.
isHere(Item, Place):- location(Item, Container), isHere(Container, Place).

put(Item, Loc):- retract(has(Item)), asserta(location(Item, Loc)).
checkPut(Item, Loc):- (room(Loc); container(Loc)), here(Here), existsHere(Loc, Here), put(Item, Loc), !.

existsHere(Place1, Place2):- Place1 == Place2, !.
existsHere(Place1, Place2):- location(Place1, Place2), !.
existsHere(Place1, Place2):- location(Place1, Place3), existsHere(Place3, Place2).

% HW1 -------------------------------------------------------------------------------------------------------------------------

connected(X,Y):- door(X,Y).
connected(X,Y):- door(Y,X).

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
printNameNC(Thing):- name(Thing, Name), write(Name), reset, !.
