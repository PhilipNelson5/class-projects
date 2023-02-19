def striped(n, p):
  indices = [[] for _ in range(p)]
  total_work = [0 for _ in range(p)]

  proc = 0
  for i in range(n):
    indices[proc].append(i)
    total_work[proc] += i + 1
    proc = (proc + 1) % p
    
  return indices, total_work

def partitioned(n, p):
  total_work = n * (n+1) / 2
  desired_work = total_work / p

  indices = [[] for _ in range(p)]
  total_work = [0 for _ in range(p)]
  
  proc = 0
  work = 0
  for i in range(n):
    indices[proc].append(i)
    total_work[proc] += i + 1
    if total_work[proc] > desired_work:
      work = 0
      proc = (proc + 1) % p
    

  return indices, total_work

def test(f, name):
  i = 0
  spread = 0
  for n in range(10, 100):
    for p in range(3, 16):
      indices, total_work = f(n, p)
      spread += max(total_work) - min(total_work)
      i += 1
  # print(total_work)
  # print(indices)
  print(name, spread/i)
  print()
  

n = 15
p = 3


  
# test(striped, 'striped')
# test(partitioned, 'partitioned')

# print(striped(n, p))
print(partitioned(n, p))
