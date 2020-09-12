---
title: 'CS 5110/6110 Program 4'
author: Philip Nelson
date: 10 April 2020
---

# Changing Number of Bidders (Linearly Distributed Bids)

More bidders pushed the price higher

![Number of Bidders](./images/simulate_num_bidders.png)

\newpage

# Changing Number of Bidders (Normally Distributed Bids)

More bidders pushed the price higher; however, the price final price per click does not end up being as high as with the linear distribution.

![Number of Bidders](./images/simulate_normal_bid_dist.png)

\newpage

# Changing the Number of Advertising Slots, Constant Number of Bidders

More slots pushed the price lower

![Number of Bidders](./images/simulate_num_items.png)

\newpage

# Changing the Number of Advertising Slots While Increasing Bidders

More items and bidders pushed the price higher
for the $i^{th}$ bidder : $0\leq i < n$.

![Number of Bidders](./images/simulate_num_bidders_items.png)

\newpage

# Changing the Number of Clicks per Slot

The price per click goes from $high$ to $low$, where $high$ goes from $500$ to $50000$ by $250$, and $low = 100$. When the $low$ remained constant and the $high$ increased, the price per click increased for the $i^{th}$ bidder : $0\leq i < n$.

![Number of Bidders](./images/simulate_clicks_per_slot_1.png)

\newpage

# Changing the Number of Clicks per Slot

The price per click goes from $high$ to $low$, where $high$ goes from $500$ to $50000$ by $25$, and $low = high-400$. When the $low$ remained a constant amount lower than the $high$, the price decreased for the $i^{th}$ bidder : $0\leq i < n$.

![Number of Bidders](./images/simulate_clicks_per_slot_2.png)

\newpage

# Changing the Number of Clicks per Slot

The price per click goes from $high$ to $low$, where $high$ goes from $500$ to $50000$ by $25$, and $low = \frac{high}{5}$. When the $low$ remained a proportional amount lower than the $high$, the price per click remained the same for the $i^{th}$ bidder.

![Number of Bidders](./images/simulate_clicks_per_slot_3.png)


