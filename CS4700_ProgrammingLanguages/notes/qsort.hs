qsort [] = []
qsort (head:tail) = qsort [less | less <- tail , less <= head] ++ [head] ++ qsort [great | great <- tail , great > head]
