Author: Philip Nelson

look(room)
The name and short description of the room
A list of connected locations with their respective short descriptions
A list of items in the room with their respective short descriptions

look(item)
The item's name and associated short description

checkLook(room)
Check the room is the current room

checkLook(item)
Check the item is in sight (in the room / in inventory)
Call study(item)

study(item)
The name of the item and it's long description

study(container)
The name of the container and its long description
a list of items it contains with their respective short descriptions
Call study(container)

checkStudy(item)
Check the item is in sight (in the room / in inventory)
Call study(item)

checkStudy(container)
Check the container is in sight (in the roomy)
Call study(container)

inventory
List the items contained in the inventory

move(Place)
retracts the current location and asserts Place

checkMove(Place)
checks if the Place is connected to the current place
calls move(Place)

take(Item)
retracts the location of the item and asserts has(Item)
list inventory

checkTake(Item)
checks if the item is in your current location or, if it is a container, that the container is in your current location
checks if the item is too heavy
calls take(Item)

put(Item, Location)
retract has(Item) then assert location(Item, Location)

checkPut(Item, Location)
checks if the location IS the current location, if it is not, it checks if the location is IN the current location
calls put(Item, Location)
