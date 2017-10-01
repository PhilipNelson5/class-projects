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

connected(X,Y) :- door(X,Y).
connected(X,Y) :- door(Y,X).

look(Place):-room(Place), yellow, write("Location:\n"), reset, descriptionShort(Place), listConnecions(Place), listItems(Place), !.
look(_).

search(Object):-container(Object), printName(Object), write(": "), long_desc(Object, Descript), write(Descript), nl, magenta, write("\nContains:\n"), reset, listContainter(Object), !.
search(Object):-location(Object, _), long_desc(Object, Descript), write(Descript), nl, fail.
search(_).

/* List contents of a container */
listContainter(Container):-location(Item, Container), descriptionLong(Item), nl, fail.
listContainter(_).

/* List connections from a room */
listConnecions(Place):-room(Place), cyan, write("\n\nConnections:\n"), reset, connected(Place,X), descriptionShort(X), nl, fail.
listConnecions(_).

/* Display the long description of a room */
descriptionLong(Place):- printName(Place), write(": "), long_desc(Place, Descript), write(Descript), !.
descriptionLong(_).

/* Display the short description of a room */
descriptionShort(Place):- printName(Place), write(": "), short_desc(Place, Descript), write(Descript), !.
descriptionShort(_).

/* List the items in a room */
listItems(Place):-room(Place), red, write("\nItems:\n"), reset, location(Item, Place), descriptionShort(Item), nl, fail.
listItems(_).

printName(Thing):- name(Thing, Name), green, write(Name), reset, !.
