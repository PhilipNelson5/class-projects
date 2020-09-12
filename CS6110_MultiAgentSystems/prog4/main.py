#!/usr/bin/env python3
from tqdm import tqdm
from math import inf
from copy import deepcopy
from typing import List, Tuple
from random import randint, choice, random
import matplotlib.pyplot as plt
import numpy as np
import os, sys, inspect

# save: bool = True
save: bool = False
# show: bool = False
show: bool = True

def vcg(clicks: List[int], bids_in: List[float], n: int) -> float:
    ''' determine the Vickrey Clark Grove payment for the nth bidder

    Keyword Arguments
    clicks  -- list of the number of clicks per slot
    bids_in -- list of the bids from the bidders
    n       -- the index of the bidder to calculate
    '''
    bids: List[float] = deepcopy(bids_in)
    bids[n] = 0.0

    #
    sum_before: float = 0.0
    for c in range(len(clicks)):
        sum_before += clicks[c] * bids[c]
    
    # remove the nth bidder
    del bids[n]

    #
    sum_after: float = 0.0
    for c in range(len(clicks)):
        sum_after += clicks[c] * bids[c]

    price = sum_after - sum_before

    return price / clicks[n]


def expected_profit(clicks: List[int], bids: List[float]) -> float:
    return sum( [ a*b for a,b in zip(clicks, bids) ] )

def run(clicks: List[int], bids: List[float]) -> List[float]:
    cost: List[float] = []

    for c in range(len(clicks)):
        cost.append( vcg(clicks, bids, c) )

    return cost


def simulate_num_bidders() -> None:
    num_trials: int = 1000
    start: int = 10
    end: int = 1000
    step: int = 10
    trials = {n:[] for n in range(start, end)}
    # clicks : List[int] = [500, 300, 100]
    clicks : List[int] = [500, 400, 300, 200, 100]

    for _ in tqdm(range(num_trials)):
        for n in range(start, end, step):
            bids: List[float] = sorted([random() for _ in range(n)], reverse = True)
            trials[n].append( run(clicks, bids) )

    avg = {n:[] for n in range(start, end)}
    for k in trials:
        avg[k] = np.mean(np.array(trials[k]), axis=0)

    plt.clf()
    for k in avg:
        plt.plot(avg[k])

    plt.title('Changing the number of bidders')
    plt.xlabel('Bidder')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'n bidders from {start} - {end} by {step}, {num_trials} trials',
            horizontalalignment='right')
    plt.xticks([x for x in range(len(clicks))])
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def simulate_normal_bid_dist() -> None:
    num_trials: int = 1000
    start: int = 10
    end: int = 1000
    step: int = 10
    trials = {n:[] for n in range(start, end)}
    clicks : List[int] = [500, 400, 300, 200, 100]

    for n in tqdm(range(start, end, step)):
        for _ in range(num_trials):
            bids: List[float] = sorted([np.random.normal(.5, .1) for _ in range(n)], reverse = True)
            trials[n].append( run(clicks, bids) )

    avg = {n:[] for n in range(start, end)}
    for k in trials:
        avg[k] = np.mean(np.array(trials[k]), axis=0)

    plt.clf()
    for k in avg:
        plt.plot(avg[k])

    plt.title('Changing the number of bidders')
    plt.xlabel('Bidder')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'n bidders from {start} - {end} by {step}, {num_trials} trials',
            horizontalalignment='right')
    plt.xticks([x for x in range(len(clicks))])
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def simulate_num_items() -> None:
    start: int = 3
    end: int = 100
    step: int = 1
    trials = {n:[] for n in range(start, end)}
    bids: List[float] = list(np.linspace(1, 0, end+1))

    for n in tqdm(range(start, end, step)):
        clicks: List[int] = [int(x) for x in list(np.linspace(500, 100, n))]
        trials[n] = run(clicks, bids)

    plt.clf()
    for k in trials:
        plt.plot(trials[k])

    plt.title('Changing the number of items')
    plt.xlabel('Bidder n')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'n items from {start} - {end} by {step}',
            horizontalalignment='right')
    plt.xticks([x for x in range(0, len(clicks)+9, 10)])
    plt.grid()
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def simulate_num_bidders_items() -> None:
    start: int = 3
    end: int = 100
    step: int = 1
    trials = {n:[] for n in range(start, end)}

    for n in tqdm(range(start, end, step)):
        clicks: List[int] = [int(x) for x in list(np.linspace(500, 100, n))]
        bids: List[float] = list(np.linspace(1, 0, n+1))
        trials[n] = run(clicks, bids)

    plt.clf()
    for k in trials:
        plt.plot(trials[k])

    plt.title('Changing Bidders and Items: Bidders = Number of Items + 1')
    plt.xlabel('Bidder n')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'n items from {start} - {end} by {step}',
            horizontalalignment='right')
    plt.xticks([x for x in range(0, len(clicks)+9, 10)])
    plt.grid()
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def simulate_clicks_per_slot_1() -> None:
    start: int = 500
    end: int = 50000
    step: int = 250
    bidders: int = 5
    trials = {}
    bids: List[float] = list(np.linspace(1, 0, bidders+5))

    for n in tqdm(range(start, end, step)):
        clicks: List[int] = [int(x) for x in list(np.linspace(n, 100, 3))]
        trials[n] = run(clicks, bids)

    plt.clf()
    for k in trials:
        plt.plot(trials[k])

    plt.title('Changing the Clicks per Slot')
    plt.xlabel('Bidder n')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'price per click {start} - {end} by {step}',
            horizontalalignment='right')
    plt.xticks([x for x in range(3)])
    plt.grid()
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def simulate_clicks_per_slot_2() -> None:
    start: int = 500
    end: int = 50000
    step: int = 25
    bidders: int = 5
    trials = {}
    bids: List[float] = list(np.linspace(1, 0, bidders+5))

    for n in tqdm(range(start, end, step)):
        clicks: List[int] = [int(x) for x in list(np.linspace(n, n-400, 3))]
        trials[n] = run(clicks, bids)

    plt.clf()
    for k in trials:
        plt.plot(trials[k])

    plt.title('Changing the Clicks per Slot')
    plt.xlabel('Bidder n')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'price per click {start} - {end} by {step}',
            horizontalalignment='right')
    plt.xticks([x for x in range(3)])
    plt.grid()
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def simulate_clicks_per_slot_3() -> None:
    start: int = 500
    end: int = 50000
    step: int = 25
    bidders: int = 5
    trials = {}
    bids: List[float] = list(np.linspace(1, 0, bidders+5))

    for n in tqdm(range(start, end, step)):
        clicks: List[int] = [int(x) for x in list(np.linspace(n, n/5, 3))]
        trials[n] = run(clicks, bids)

    plt.clf()
    for k in trials:
        plt.plot(trials[k])

    plt.title('Changing the Clicks per Slot')
    plt.xlabel('Bidder n')
    plt.ylabel('Price Per Click')
    plt.figtext(
            0.99, 0.01,
            f'price per click {start} - {end} by {step}',
            horizontalalignment='right')
    plt.xticks([x for x in range(3)])
    plt.grid()
    if save: plt.savefig(f'images/{inspect.stack()[0][3]}.png')
    if show: plt.show()


def main() -> None:
    clicks : List[int] = [500, 300, 100]
    bids: List[float] = [.5, .4, .3, .2, .1]

    simulate_num_bidders()
    simulate_normal_bid_dist()
    simulate_num_items()
    simulate_num_bidders_items()
    simulate_clicks_per_slot_1()
    simulate_clicks_per_slot_2()
    simulate_clicks_per_slot_3()
    # print(run(clicks, bids))

if __name__ == '__main__':
    main()
