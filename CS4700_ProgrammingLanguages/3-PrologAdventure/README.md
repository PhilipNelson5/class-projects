### Author: Philip Nelson

To play, load logic.pl in a prolog interpreter.

For best results, use a terminal that supports colors

---

`look.`
* Look at the current room

`look(room).`
* The name and short description of the room
* A list of connected locations with their respective short descriptions
* A list of items in the room with their respective short descriptions

`look(item).`
* The item's name and associated short description

`checkLook(room).`
* Check the room is the current room

`checkLook(item).`
* Check the item is in sight (in the room / in inventory)
* Call `study(item)`

`study(item).`
* The name of the item and it's long description

`study(container).`
* The name of the container and its long description
* A list of items it contains with their respective short descriptions
* Call `study(container)`

`checkStudy(item).`
* Check the item is in sight (in the room / in inventory)
* Call `study(item)`

`checkStudy(container).`
* Check the container is in sight (in the roomy)
* Call `study(container)`

`inventory.`
* List the items contained in the inventory

`move(Place).`
* Retracts the current location and asserts Place

`checkMove(Place).`
* Checks if the Place is connected to the current place
* check for a puzzle in the current room
* Calls `move(Place)`

`take(Item).`
* Retracts the location of the item and asserts has(Item)
* List inventory

`checkTake(Item).`
* Checks if the item is in your current location or, if it is a container, that the container is in your current location
* Checks if the item is too heavy
* Calls `take(Item)`

`put(Item, Location).`
* Retract `has(Item)` then assert `location(Item, Location)`

`checkPut(Item, Location).`
* Checks if the location IS the current location, if it is not, it checks if the location is IN the current location
* Calls `put(Item, Location)`

`give(Item)`
* asserts that the player has the item

`transfer(Disk, Pylon_f)`
* moves the disk from pylon1 to pylon2

`checkTransfer(Disk, Pylon_f)`
* checks the player is in the secret lab
* checks the move is valid as dictated by the rules of the Towers of Hanoi
* calls `transfer(Disk, Pylon_f)`

`make(Product)`
* retract all ingredients from inventory
* asserts `has(Product)`

`checkMake(Product)`
* checks that the player has all the ingredients
* checks that the necessary equipment is present
* calls `make(product)`

`printTowers`
* prints the pylons and the disks in a neat visual ASCII form

`win`
* checks that the disks are all on pylon_c
* displays a congratulatory message
