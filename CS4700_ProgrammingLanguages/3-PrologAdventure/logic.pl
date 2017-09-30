:-include(adventure).
connected(X,Y) :- door(X,Y).
connected(X,Y) :- door(Y,X).

look(Place):-room(Place), descriptionShort(Place), listConnecions(Place), listItems(Place), !.
look(_).

search(Object):-container(Object), printName(Object), write(": "), long_desc(Object, Descript), write(Descript), nl, write("contains:\n"), listContainter(Object), !.
search(Object):-location(Object, _), long_desc(Object, Descript), write(Descript), nl, fail.
search(_).

/* List contents of a container */
listContainter(Container):-location(Item, Container), descriptionLong(Item), nl, fail.
listContainter(_).

/* List connections from a room */
listConnecions(Place):-room(Place), write("\n\nConnections:\n"), connected(Place,X), descriptionShort(X), nl, fail.
listConnecions(_).

/* Display the long description of a room */
descriptionLong(Place):- printName(Place), write(": "), long_desc(Place, Descript), write(Descript), !.
descriptionLong(_).

/* Display the short description of a room */
descriptionShort(Place):- printName(Place), write(": "), short_desc(Place, Descript), write(Descript), !.
descriptionShort(_).

/* List the items in a room */
listItems(Place):-room(Place), write("\nItems:\n"), location(Item, Place), printName(Item), nl, fail.
listItems(_).

printName(Thing):- name(Thing, Name), write(Name), !.
