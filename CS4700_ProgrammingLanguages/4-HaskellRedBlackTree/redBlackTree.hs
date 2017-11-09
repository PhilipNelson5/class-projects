data Color = Red | Black deriving (Show)
data Tree a = Nill | Node Color (Tree a) a (Tree a) deriving (Show)

singleton :: a -> Tree a
singleton x = Node Red Nill x Nill

insert :: (Ord a) => a -> Tree a -> Tree a
insert x Nill = singleton x
insert x (Node c l v r)
  | x == v = Node c l v r
  | x < v  = balance $ Node c (insert x l) v r
  | x > v  = balance $ Node c l v (insert x r)

balance :: Tree a -> Tree a
balance (Node Black (Node Red (Node Red a x b) y c ) z d) = (Node Red (Node Black a x b) y (Node Black c z d))
balance (Node Black (Node Red a x (Node Red b y c)) z d) = (Node Red (Node Black a x b) y (Node Black c z d))
balance (Node Black a x (Node Red (Node Red b y c) z d)) = (Node Red (Node Black a x b) y (Node Black c z d))
balance (Node Black a x (Node Red b y (Node Red c z d))) = (Node Red (Node Black a x b) y (Node Black c z d))
balance t = t

makeBlack :: Tree a -> Tree a
makeBlack(Node Red l v r) = Node Black l v r
makeBlack t = t

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert t x = makeBlack(insert t x)

draw :: Show a => Tree a -> Int -> String
draw Nill x = tabs x ++ "Nill\n"
draw (Node Black r v l) x = (draw l (x+1)) ++ (tabs x) ++ "B " ++ show v ++ "\n" ++ (draw r (x+1))
draw (Node Red r v l) x = (draw l (x+1)) ++ (tabs x) ++ "R " ++ show v ++ "\n" ++ (draw r (x+1))

tabs :: Int -> String
tabs x = foldr (++) "" (take x $ repeat "\t")

drawt :: Show a => Tree a -> IO ()
drawt t = putStr $ draw t 0
