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
balance (Node Black (Node Red (Node Red a x b) y c ) z d) = (Node Red (Node Black a x b) y (Node Black c z d))
balance (Node Black (Node Red a x (Node Red b y c)) z d) = (Node Red (Node Black a x b) y (Node Black c z d))
balance (Node Black a x (Node Red (Node Red b y c) z d)) = (Node Red (Node Black a x b) y (Node Black c z d))
balance (Node Black a x (Node Red b y (Node Red c z d))) = (Node Red (Node Black a x b) y (Node Black c z d))
balance t = t

makeBlack(Node Red l v r) = Node Black l v r
makeBlack t = t

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert t x = makeBlack(insert t x)

draw :: Tree a -> Int -> String
draw Nill x = foldr (++) "" (take x $ repeat "\t") ++ "Nill\n"
draw (Node Black r v l) x = (draw r (x+1)) ++ (foldr (++) "" (take x $ repeat "\t") ++ "B\n") ++ (draw l (x+1))
draw (Node Red r v l) x = (draw r (x+1)) ++ (foldr (++) "" (take x $ repeat "\t") ++ "R\n") ++ (draw l (x+1))
--draw (Node Black r v l) x = (draw r (x+1)) ++ ("R" ++ (show v)) ++ (draw l (x+1))

drawTree :: Tree a -> IO ()
drawTree t = putStr $ draw t 0 ++ "\n"
