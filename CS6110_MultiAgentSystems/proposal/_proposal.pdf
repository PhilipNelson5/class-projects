---
title: Determining the Effect of Influencing Agents on Flock Density
author: |
  | Philip Nelson
  | Department of Computer Science
  | Utah State University, Logan, UT 84321 USA
  | philip.nelson@aggiemail.usu.edu
date: 12th March 2020
fontsize: 11pt
documentclass: article
classoption:
- twocolumn
geometry:
- left=1in
- right=1in
- top=1in
- bottom=1in
#header-includes: |
  #\usepackage{mathtools}
---

# Abstract

Flocking is a multi agent behavior exhibited naturally by many different animals including birds, fish, insects, herds of land animals and more. Although flocking might look like a complex orchestration of tens to millions of individuals, it is the simple result of each individual reacting to its own local environment. Previous work has been done on influencing flocks through carefully placed "influencing agents". [2] Influencing agents can be used to direct a flock of agents despite only controlling a small fraction of the flock. In this paper, I intend to look at the methods previously proposed and measure how they impact flock density. I will look at random, grid, border, and graph placement as described by Dr. Katie Genter. [2]

# Introduction
Flocking behavior emerges from a set of simple governing rules followed by each agent. In our experiment, I will use the boid flocking model described by Craig Reynolds [1]. In his paper, Reynolds gives the following three rules.

1. Collision Avoidance: avoid collisions with nearby flockmates
2. Velocity Matching: attempt to match velocity with nearby flockmates
3. Flock Centering: attempt to stay close to nearby flockmates 

From these simple rules, much like Conway's Game of Life, can emerge complex and lifelike simulations. Each member of the flock is able to see nearby flockmates within a radius $r$. Scientists have used high speed cameras to observe and model flocks of birds which has confirmed that these basic rules generally hold with some caveats. [4] The flock centering rule usually applies to the nearest 5-10 neighbors of each flock mate and is independent of the distance of these neighbors from the flock member. Additionally there is a stronger tendency to stay close to flockmates which are to the side of the flock member as opposed to in front, behind, above or below.

Motivation for influencing a flock could be for their protection and the protection of human infrastructure. This technique could be used to place a few influencing agents, in the form of robotic flock members, in a flock that is heading towards an airport or wind farm in order to guide them safely around the hazard in a minimally invasive way. I think it would be interesting measure which flock placement method leads to a more compact or sparse flock in addition to which method achieves the best flock control.

# Background and Related Work
The four placement methods I will use are described by Dr. Genter in her dissertation: random placement, grid placement, border placement, and graph placement. [3] Random placement places $k$ influencing agents randomly within the flock. Grid placement uses an evenly spaced grid to place $k$ influencing agents throughout a flock. The border placement method puts agents around the perimeter such that at most $\left\lceil\frac{k}{4}\right\rceil$ agents are placed on a side. 

These placement methods are straight forward and Genter shows that they have some success influencing the flock. The more interesting method for influencing a flock is graph placement. In this method, a graph is used to represent the flock. Each natural, non-influencing, member of the flock is added to the graph and undirected edges are placed between members to represent the Cartesian distance between them. Since each member of the flock has a limited visibility radius, $r$, influencing agents are placed at the midpoint between flock members where the edge between them is at most $2r$. This way, the influencing agent can influence both neighbors. In the case that there are no $2r$ edges, the agents are place such that they are in the vicinity of one other flock member.

# Proposed Work

I am proposing to develop a metric to measure flock density and simulate four placement methods: grid, border, and graph, in order to determine the effect of the influencing agents on the flock's density. I plan to build a simulation based on the boid flock model and implement agents who follow the boid rules and agents that are externally controlled. These influencing agents will have knowledge of the other members of the flock and will be seen by the flock as regular members. I will then use this simulation to determine which placement has the greatest effect on flock density. This first part will look at the passive effect of the influencing agents. Then I would like to determine if or how the influencing agents are able to manipulate the flock density. Using the same density metric I will test different strategies to artificially expand or contract a flock.

I expect to see that the border placement method has the greatest capacity for compacting the flock and increasing flock density. Conversely, I expect that the graph placement method has the greatest capacity for expanding the flock and decreasing flock density.

# References

[1] C. W. Reynolds. Flocks, herds and schools: A distributed
behavioral model. SIGGRAPH, 21:25–34, August 1987.

[2] K. Genter, S. Zhang, and P. Stone. Determining placements of influencing agents in a flock.
In Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent
Systems (AAMAS’15), pages 247–255. International Foundation for Autonomous Agents and
Multiagent Systems, May 2015.

[3] K. Genter. Fly with Me: Algorithms and Methods for Influencing a Flock [Doctoral dissertation] The University of Texas at Austin. repositories.lib.utexas.edu, August 2017

[4] T. Feder. "Statistical physics is for the birds". Physics Today. 60 (10): 28–30. doi:10.1063/1.2800090, October 2007
