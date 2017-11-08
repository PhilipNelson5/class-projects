data Color = Red | Black deriving (Show)
data Tree a = Nill | Node Color (Tree a) a (Tree a) deriving (Show)

singleton :: a -> Tree a
singleton x = Node Red Nill x Nill

insert :: (Ord a) => a -> Tree a -> Tree a
insert x Nill = singleton x
insert x (Node c left a right)
  | x == a = Node c left a right
  | x < a  = Node c (balance(insert x left)) a right
  | x > a  = Node c left a (balance(insert x right))

balance :: Tree a -> Tree a
balance (Node Black (Node Red a x (Node Red b y c)) z d) = (Node Red (Node Black a x b) y (Node Black c z d))
balance t = t

makeBlack(Node Red l v r) = Node Black l v r
makeBlack t = t

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert t x = makeBlack(insert t x)

isElem :: (Ord a) => a -> Tree a -> Bool
isElem x Nill = False
isElem x (Node c left a right)
  | x == a = True
  | x < a = isElem x left
  | x > a = isElem x right

