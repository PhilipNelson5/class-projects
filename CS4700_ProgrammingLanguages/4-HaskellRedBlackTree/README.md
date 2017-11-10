# Persistent Red Black Tree in Haskell

Functionality Testing:
- `t = foldr treeInsert Nill [list]` , will build a tree and assign it to `t`.
- `display t` , will pretty print the tree `t`.
- `height t` , will return the height of the tree `t`.

Alternatively:
- `foldr treeInsert Nill [list]` , will build a tree and "show" it.
- `display $ foldr treeInsert Nill [list]` , will pretty print the tree.
- `height $ foldr treeInsert Nill [list]` , will return the height of the tree.
