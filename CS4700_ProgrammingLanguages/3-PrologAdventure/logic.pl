:-include(adventure).
:-include(words).
:-dynamic theEnd/1.
:-dynamic devMode/1.
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

theEnd(no).
devMode(no).

% HW4 -------------------------------------------------------------------------------------------------------------------------

play:-clear, green, welcome, blue, prologAdventure, white, intro, reset, repeat, write("\nWhat do you want to do? "), read_words(W), clear, parse(C,W), do(C), win, !.

parse([V],Input):- verb(V,Input-[]).
parse([V,O],Input):- verb(V,Input-NounPhrase), noun(O, NounPhrase-[]).
parse([V,D,P],Input):- verb(V,Input-NounPhrase), noun(D, NounPhrase-PylonPhrase), noun(P, PylonPhrase-[]).

verb(quit,["quit"|X]-X).
verb(quit,["exit"|X]-X).

verb(checkLook, ["look","at"|X]-X).
verb(checkLook, ["look","in"|X]-X).
verb(checkLook, ["look","at","the"|X]-X).
verb(checkLook, ["look","in","the"|X]-X).

verb(lookHere, ["look"|X]-X).
verb(lookHere, ["look","here"|X]-X).

verb(checkStudy, ["study"|X]-X).
verb(checkStudy, ["study","the"|X]-X).

verb(dev, ["dev"|X]-X).
verb(move, ["warp"|X]-X):-devMode(yes).

verb(inventory, ["inv"|X]-X).
verb(inventory, ["inventory"|X]-X).
verb(inventory, ["look","inventory"|X]-X).
verb(inventory, ["look","at","inventory"|X]-X).

verb(checkMove, ["go"|X]-X).
verb(checkMove, ["go","to"|X]-X).
verb(checkMove, ["go","to","the"|X]-X).
verb(checkMove, ["walk"|X]-X).
verb(checkMove, ["walk","to"|X]-X).
verb(checkMove, ["walk","to","the"|X]-X).

verb(checkTake, ["take"|X]-X).
verb(checkTake, ["take the"|X]-X).
verb(checkTake, ["pickup"|X]-X).
verb(checkTake, ["pickup the"|X]-X).
verb(checkTake, ["grab"|X]-X).
verb(checkTake, ["grab the"|X]-X).

verb(transfer, ["transfer"|X]-X).

noun(agricultural_science,["agricultural sciences"|X]-X).
noun(agricultural_science,["ag science"|X]-X).
noun(animal_science,["animal","sciences"|X]-X).
noun(animal_science,["animal","science"|X]-X).
noun(avenue,["avenue"|X]-X).
noun(bedroom,["bedroom"|X]-X).
noun(bedroom_closet,["closet"|X]-X):-here(bedroom).
noun(bone,["dragon","bone"|X]-X).
noun(book_a,["corpus","hermiticum"|X]-X).
noun(book_b,["war","and","peace"|X]-X).
noun(book_c,["great","expectations"|X]-X).
noun(bunsen_burner,["bunsen","burner"|X]-X).
noun(charged_bone,["chunk","of","dragon","bone"|X]-X).
noun(chemistry_lab,["chemistry","lab"|X]-X).
noun(chemistry_lab,["lab"|X]-X):-here(eslc_north).
noun(clean_clothes, ["your","clothes"|X]-X).
noun(clean_clothes, ["clean","clothes"|X]-X).
noun(closet,["equipment","closet"|X]-X).
noun(closet,["closet"|X]-X):-here(eslc_south).
noun(coat,["lab","coat"|X]-X).
noun(coat,["coat"|X]-X).
noun(common_room,["dorm","common","room"|X]-X).
noun(common_room,["common","room"|X]-X).
noun(computer_lab,["student","computer lab"|X]-X).
noun(computer_lab,["computer","lab"|X]-X).
noun(dirty_clothes, ["your","dirty","clothes"|X]-X).
noun(dirty_clothes, ["dirty","clothes"|X]-X).
noun(elevator,["elevator"|X]-X).
noun(engr,["engr"|X]-X).
noun(engr,["ENGR"|X]-X).
noun(eslc_north,["eccels","science","learning","center"|X]-X).
noun(eslc_north,["eslc"|X]-X):-here(chemistry_lab).
noun(eslc_north,["eslc"|X]-X):-here(tsc_patio).
noun(eslc_north,["eslc"|X]-X):-here(eslc_south).
noun(eslc_south,["eccels","science","learning","center"|X]-X).
noun(eslc_south,["eslc"|X]-X):-here(eslc_north).
noun(eslc_south,["eslc"|X]-X):-here(quad).
noun(figurine,["alien","figurine"|X]-X).
noun(figurine,["figurine"|X]-X).
noun(flask,["glass","flask"|X]-X).
noun(flask,["flask"|X]-X).
noun(fly,["dead","fly"|X]-X).
noun(fly,["fly"|X]-X).
noun(gas_lab, ["get","away","special","lab"|X]-X).
noun(gas_lab, ["gas","lab"|X]-X).
noun(geology_building,["geology","building"|X]-X).
noun(goggles,["dark","safety goggles"|X]-X).
noun(goggles,["safety","goggles"|X]-X).
noun(goggles,["goggles"|X]-X).
noun(green_beam,["grean","beam","enclosure"|X]-X).
noun(hall,["hallway"|X]-X).
noun(hall,["hall"|X]-X).
noun(hub,["hub"|X]-X).
noun(ice_cream,["aggie","creamery"|X]-X).
noun(ice_cream,["aggie","ice","cream"|X]-X).
noun(key,["key"|X]-X).
noun(kitchen,["kitchen"|X]-X).
noun(large_disk,["large","energy","disk"|X]-X).
noun(large_disk,["large","disk"|X]-X).
noun(laser,["laser","array"|X]-X).
noun(laser_lab,["laser","lab"|X]-X).
noun(laser_lab,["lab"|X]-X):-here(ser_2nd_floor).
noun(library,["merill-caizer","library"|X]-X).
noun(library,["library"|X]-X).
noun(lost_homework,["some","student's","lost","geometry","homework"|X]-X).
noun(lost_homework,["student's","lost","geometry","homework"|X]-X).
noun(lost_homework,["lost","geometry","homework"|X]-X).
noun(lost_homework,["geometry","homework"|X]-X).
noun(medium_disk,["medium","energy","disk"|X]-X).
noun(medium_disk,["medium","disk"|X]-X).
noun(movie,["men","in","black"|X]-X).
noun(note,["note"|X]-X).
noun(note1_gas,["note","1","gas"|X]-X).
noun(observatory,["observatory"|X]-X).
noun(old_main,["old","main"|X]-X).
noun(plaza,["engineering","plaza"|X]-X).
noun(plaza,["plaza"|X]-X).
noun(potion,["potion"|X]-X).
noun(pylon_a,["red","pylon"|X]-X).
noun(pylon_b,["blue","pylon"|X]-X).
noun(pylon_c,["green","pylon"|X]-X).
noun(quad,["quad"|X]-X).
noun(recipe,["alchemical","recipe"|X]-X).
noun(roof,["roof","of","the","ser","building"|X]-X).
noun(roof,["ser","roof"|X]-X).
noun(roof,["roof"|X]-X):-here(elevator).
noun(roof,["roof"|X]-X):-here(green_beam).
noun(roof,["roof"|X]-X):-here(observatory).
noun(roommate_room,["your","dormmate's","room"|X]-X).
noun(roommate_room,["dormmate's","room"|X]-X).
noun(secret_lab,["dr.","sundberg's","secret","lab"|X]-X).
noun(secret_lab,["sundberg's","secret","lab"|X]-X).
noun(secret_lab,["secret","lab"|X]-X).
noun(ser_1st_floor,["1st","Floor","of","ser","Building"|X]-X).
noun(ser_1st_floor,["ser","1st","floor"|X]-X).
noun(ser_1st_floor,["1st","floor"|X]-X):-here(elevator).
noun(ser_1st_floor,["ser"|X]-X):-here(plaza).
noun(ser_2nd_floor,["2nd","floor","of","ser","building"|X]-X).
noun(ser_2nd_floor,["ser","2nd","floor"|X]-X).
noun(ser_2nd_floor,["2nd","floor"|X]-X):-here(elevator).
noun(ser_2nd_floor,["2nd","floor"|X]-X):-here(ser_conference).
noun(ser_2nd_floor,["2nd","floor"|X]-X):-here(laser_lab).
noun(ser_basement,["basement","of","the","ser","building"|X]-X).
noun(ser_basement,["ser","basement"|X]-X).
noun(ser_basement,["basement"|X]-X):-here(elevator).
noun(ser_basement,["basement"|X]-X):-here(gas_lab).
noun(ser_conference,["ser","conference","room"|X]-X).
noun(ser_conference,["conference","room"|X]-X):-here(ser_2nd_floor).
noun(small_disk,["small","energy","disk"|X]-X).
noun(small_disk,["small","disk"|X]-X).
noun(special_collections,["special","collections","room"|X]-X).
noun(special_collections,["special","collections"|X]-X).
noun(tsc,["taggart","student","center"|X]-X).
noun(tsc,["tsc"|X]-X).
noun(tsc_patio,["patio","of","the","tsc"|X]-X).
noun(tsc_patio,["tsc","patio"|X]-X).
noun(tunnels_east,["underground","tunnels","east"|X]-X).
noun(tunnels_east,["tunnels","east"|X]-X).
noun(tunnels_north,["underground","tunnels","north"|X]-X).
noun(tunnels_north,["tunnels","north"|X]-X).
noun(tunnels_west,["underground","tunnels","west"|X]-X).
noun(tunnels_west,["tunnels","west"|X]-X).

do(C):- CMD =.. C, CMD, !.

quit:- write("Bye"), nl, asserta(theEnd(yes)).

dev:- write("Dev Mode Enabled"), nl, asserta(devMode(yes)).

outside(avenue).
outside(plaza).
outside(quad).
outside(roof).

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
checkMake(Product):- create_recipe(Device, List, Product), here(Here), location(Device, Here), hasAll(List), make(Product).

hasAll([]).
hasAll([H|T]):- has(H), hasAll(T).

use([]).
use([H|T]):- has(H), retract(has(H)), use(T).

win:- theEnd(yes), !.
win:- location(large_disk, pylon_c), location(medium_disk, pylon_c), location(small_disk, pylon_c), nl, write("You foiled the evil Dr. Sundberg!"),nl, nl.

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

hanoiLables:- red, write("¯¯¯¯¯¯¯¯¯"), blue, write("¯¯¯¯¯¯¯¯¯"), green, write("¯¯¯¯¯¯¯¯¯"), reset.

% HW2 -------------------------------------------------------------------------------------------------------------------------

lookHere:- here(X), look(X), !.
lookHere(_).

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
