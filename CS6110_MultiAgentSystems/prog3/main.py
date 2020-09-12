#!/usr/bin/env python3
from math import inf
from copy import deepcopy
from typing import List, Tuple
from random import randint, choice
import os, sys
Matrix = List[List[int]]


A = ord('A')
menu = """
┌───────────────────────────────────┐
│Chose an option                    │
│1) run data set 1                  │
│2) run data set 2                  │
│3) manual entry                    │
│4) random                          │
│5) specify candidate order (B > D) │
│6) offset each vote group one place│
└───────────────────────────────────┘
"""

class Node():
    def __init__(self, name: int):
        self.adj: List[Node] = []
        self.score: float = 0.0
        self.name: int = name


def clear() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
    # pass


def showMenu(lb: int, ub: int) -> int:
    resp = -1
    while(resp < lb or resp > ub):
        clear()
        print(menu)
        try:
            resp = int(input(f"enter {lb} - {ub}: "))
        except KeyboardInterrupt:
            sys.exit()
        except:
            pass
    return resp


def getPositiveInt(prompt: str) -> int:
    while True:
        try:
            resp =  int(input(prompt))
            if resp <= 0 :
                print("try again")
                continue
            return resp
        except KeyboardInterrupt:
            sys.exit()
        except:
            print("try again")

def chr_to_ord(c: str) -> int:
    n = ord(c)
    if n >= 65 and n <= 90: # Upper Case
        return n - ord("A")
    if n >= 97 and n <= 122:# Lower Case
        return n - ord("a")
    raise ValueError


def getParetoPair(num_candidates: int) -> Tuple[int, int]:
    while True:
        try:
            resp =  input("enter a pair in the form A > B: ")
            if len(resp) != 3 and len(resp) != 5:
                print("try again")
                continue

            a = chr_to_ord(resp[0])
            b = chr_to_ord(resp[-1])
            if a > num_candidates or b > num_candidates:
                print("Max candidate is", chr(A+num_candidates-1))
                continue

            if resp[len(resp)//2] == '>':
                return (a,b)
            else:
                return (b,a)

        except KeyboardInterrupt:
            sys.exit()
        except:
            print("try again")


def read(file) -> List[List[int]]:
    candidates = []
    f = open(file)
    count = [int(x) for x in f.readline().split(',')]
    for line in f: candidates.append([int(x) for x in line.strip().split(',')])

    voters = [[0 for _ in range(len(candidates))] for _ in range(len(candidates[0]))]
    for candidate in range(len(candidates)):
        for voter in range(len(candidates[0])):
            voters[voter][candidate] = candidates[candidate][voter]

    rank_choice = []
    for voter in voters:
        rank_choice.append([voter.index(x) for x in range(1, len(candidates)+1)])

    return candidates, rank_choice, count


def printVoters(voters: List[List[int]], count: List[int], num_candidates: int, num_vote_groups: int) -> None:
    print(" "*8, end="")
    for i in range(num_candidates): print(chr(i+A), "", end="")
    print()

    for v in range(num_vote_groups):
        if v > len(count)-1: print(chr(v+A), "(--): ", end="")
        else: print(chr(v+A), f"({count[v]}){' ' if count[v] < 10 else ''}: ", end="")

        for c in range(num_candidates):
            if v > len(voters)-1 or c > len(voters[v])-1:
                print ('- ', end="")
            else: print(voters[v][c], "", end="")

        print()


def userEntry(num_candidates: int, num_vote_groups: int):
    voters = []
    count = []

    clear()
    for v in range(num_vote_groups):
        voters.append([])

        good = False
        while not good:
            printVoters(voters, count, num_candidates, num_vote_groups)
            print()
            try:
                n = int(input(f"How many voters in group {chr(v+A)}? "))
                if n >= 1: good = True
            except KeyboardInterrupt: sys.exit()
            except: pass
            clear()
        count.append(n)

        places = [x for x in range(1, num_candidates+1)]
        for c in range(num_candidates):
            good = False
            while not good:
                printVoters(voters, count, num_candidates, num_vote_groups)
                print()
                try:
                    n = int(input(f"Rank candidate {chr(c+A)} from 1 to {num_candidates}: "))
                    if n not in voters[-1] and n <= num_candidates and n > 0: good = True
                except KeyboardInterrupt: sys.exit()
                except: pass
                clear()

            voters[-1].append(n)

    rank_choice = []
    for voter in voters:
        rank_choice.append([voter.index(x) for x in range(1, num_candidates+1)])

    return rank_choice, count



def randomInput(num_candidates: int, num_vote_groups: int, paretoPair: Tuple[int, int] = None):
    vote_group_max_size: int = 10
    voters = []
    count = []
    for v in range(num_vote_groups):
        voters.append([])
        count.append(randint(1, vote_group_max_size))
        places = [x for x in range(1, num_candidates+1)]
        for _ in range(num_candidates):
            place = choice(places)
            places.remove(place)
            voters[-1].append(place)

    if paretoPair != None:
        for voter in voters:
            a = voter[paretoPair[0]]-1
            b = voter[paretoPair[1]]-1
            if b < a:
                voter[paretoPair[1]], voter[paretoPair[0]] = voter[paretoPair[0]], voter[paretoPair[1]]

    rank_choice = []
    for voter in voters:
        rank_choice.append([voter.index(x) for x in range(1, num_candidates+1)])

    return rank_choice, count


def randomSpiral(num_candidates: int, num_vote_groups: int):
    vote_group_max_size: int = 10
    count = []
    rank_choice = []
    for v in range(num_vote_groups):
        # count.append(randint(1, vote_group_max_size))
        count.append(1)
        rank_choice.append([])
        for c in range(num_candidates):
            rank_choice[-1].append((num_candidates-v+c)%num_candidates)

    return rank_choice, count


def paretoEfficient(paretoPair: Tuple[int, int], num_candidates: int, num_vote_groups: int):
    vote_group_max_size: int = 10
    voters = []
    count = []
    for v in range(num_vote_groups):
        voters.append([])
        count.append(randint(1, vote_group_max_size))
        places = [x for x in range(1, num_candidates+1)]
        for _ in range(num_candidates):
            place = choice(places)
            places.remove(place)
            voters[-1].append(place)


    rank_choice = []
    for voter in voters:
        rank_choice.append([voter.index(x) for x in range(1, num_candidates+1)])

    return rank_choice, count



def print_ranked_choice(rank_choice: Matrix, count: List[int]) -> None:
    v = 0
    width = (len(rank_choice[0])*4+7)
    if width - 8 > 0:
        print(' '*((width-8)//2), end="")
    print("rank choice")
    print("┌"+("─"*width), end="")
    print("┐")
    for voter in rank_choice:
        print(f'│ {chr(A+v)} ({count[v]}){" " if count[v] < 10 else ""}: ',  end="")
        v += 1
        for i in range(len(voter)-1):
            print(chr(A+voter[i]), '> ', end="")
        print(chr(A+voter[-1]), '│')
    print("└"+("─"*width), end="")
    print("┘")


def print_majority_graph(candidates: List[Node]) -> None:
    for can in candidates:
        print(chr(A+can.name), f"({can.score}): ", end="")
        for opp in can.adj:
            print(chr(A+opp.name), "", end="")
        print()
    print()


def bucklin(rank_choice: Matrix, count: List[int]) -> Tuple[int, int]:
    num_vote_groups = len(rank_choice)
    num_candidates = len(rank_choice[0])
    num_voters = sum(count)

    for k in range(1, num_candidates+1):
        candidate_votes = [0 for _ in range(num_candidates+1)]
        for v in range(num_vote_groups):
            for i in range(k):
                candidate_votes[rank_choice[v][i]] += count[v]
        majority = num_voters / 2
        # print(candidate_votes)
        # print("majority",majority)
        # print("max",max(candidate_votes))
        max_votes = max(candidate_votes)
        if(max_votes > majority):
            # print(max(candidate_votes))
            return candidate_votes.index(max_votes), k


def pairwiseElection(rank_choice: Matrix, count: List[int], c1: int, c2: int) -> Tuple[int, int]:
    c1votes = 0
    c2votes = 0
    for v in range(len(rank_choice)):
        c1idx = rank_choice[v].index(c1)
        c2idx = rank_choice[v].index(c2)

        if c1idx < c2idx:
            c1votes += count[v]
        else:
            c2votes += count[v]

    return c1votes, c2votes


def makeMajorityGraph(rank_choice: Matrix, count: List[int]) -> List[Node]:
    num_vote_groups = len(rank_choice)
    num_candidates = len(rank_choice[0])
    graph: List[Node] = [Node(x) for x in range(num_candidates)]

    for c1 in range(num_candidates):
        for c2 in range(c1+1, num_candidates):
            c1votes, c2votes = pairwiseElection(rank_choice, count, c1, c2)
            # print(chr(A+c1), "vs", chr(A+c2), ": ", end="")
            # print(c1votes, '-', c2votes)

            if c1votes > c2votes:
                graph[c1].adj.append(graph[c2])
                graph[c1].score += 1
            elif c1votes < c2votes:
                graph[c2].adj.append(graph[c1])
                graph[c2].score += 1
            else:
                graph[c1].adj.append(graph[c2])
                graph[c1].score += 0.5
                graph[c2].adj.append(graph[c1])
                graph[c2].score += 0.5

    return graph


def secondOrderCopeland(graph: List[Node]) -> List[int]:
    scores = {}
    for can in graph:
        scores[can.name] = 0
        for opp in can.adj:
            scores[can.name] += opp.score

    # print()
    # print_majority_graph(graph)
    # print(scores, end='\n\n')

    return sorted(scores, key=scores.get, reverse=True)


def singleTransferableVote(rank_choice_in: Matrix, count: List[int]) -> List[int]:
    rank_choice = deepcopy(rank_choice_in)
    num_vote_groups = len(rank_choice)
    num_candidates = len(rank_choice[0])
    ranking = []
    plurality = {x:0 for x in range(num_candidates)}
    for c in range(num_candidates):
        for can in plurality: plurality[can] = 0
        for v in range(num_vote_groups):
            plurality[rank_choice[v][0]] += count[v]

        loser = min(plurality, key=plurality.get)
        plurality.pop(loser)
        
        # print("loser:", chr(A+loser))
        # print_ranked_choice(rank_choice, count)
        for vote_group in rank_choice:
            vote_group.remove(loser)
        ranking.append(loser)

    ranking.reverse()
    return ranking


def runElection(graph: List[Node], rank_choice: Matrix, count: List[int]) -> None:
    num_candidates = len(rank_choice[0])
    print_ranked_choice(rank_choice, count)

    print('\t', end="")
    for x in range(num_candidates): print (x+1, "", end="")
    print()
    print('\t', end="")
    D = '━'
    print(f'{D}{D}'*(num_candidates-1)+D)

    """ Bucklin """
    winner, k = bucklin(rank_choice, count)
    print("Bucklin\t", end="")
    print(chr(A+winner), "- "*(num_candidates-1),"k =",k)

    """ 2nd Order Copeland """
    print("2nd OC\t", end="")
    soc_ranking = secondOrderCopeland(graph)
    for c in soc_ranking: print(chr(A+c), "", end = "")
    print()

    """ Single Transferable Vote """
    stv_ranking = singleTransferableVote(rank_choice, count)
    print("STV\t", end="")
    for c in stv_ranking: print(chr(A+c), "", end = "")
    print()


def main() -> None:
    
    while True:
        resp = showMenu(0, 6)
        if resp == 0:
            break
        elif resp == 1:
            candidates, rank_choice, count = read("dataset1.csv")
        elif resp == 2:
            candidates, rank_choice, count = read("dataset2.csv")
        elif resp == 3:
            rank_choice, count = userEntry(
                    getPositiveInt("Enter the number of candidates: "),
                    getPositiveInt("Enter the number of voter groups: "))
        elif resp == 4:
            rank_choice, count = randomInput(
                    getPositiveInt("Enter the number of candidates: "),
                    getPositiveInt("Enter the number of voter groups: "))
        elif resp == 5:
            num_candidates = getPositiveInt("Enter the number of candidates: ")
            num_vote_groups = getPositiveInt("Enter the number of voter groups: ")
            paretoPair = getParetoPair(num_candidates)
            rank_choice, count = randomInput(
                    num_candidates, num_vote_groups, paretoPair)

            print()
            width = (len(rank_choice[0])*4+7)
            if width - 8 > 0:
                print(' '*((width-8)//2), end="")
            print(chr(paretoPair[0]+A), ">", chr(paretoPair[1]+A))
        elif resp == 6:
            rank_choice, count = randomSpiral(
                    getPositiveInt("Enter the number of candidates: "),
                    getPositiveInt("Enter the number of voter groups: "))

        graph = makeMajorityGraph(rank_choice, count)
        runElection(graph, rank_choice, count)
        # clear()

        input("\nPress any key to return to the menu...")


if __name__ == '__main__':
    main()
