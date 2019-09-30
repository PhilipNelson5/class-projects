---
title: IVP Test
math: true
layout: default
---

{% include mathjax.html %}

# Iterative Methods Summary

Iterative methods studied in assignment 5 are examples iterative methods used to solve initial value problems.
Each method has it's strengths and weaknesses, there is best method. Each has different conditions for convergence, but some have better overall performance.

The Explicit Euler Method is a very simple technique compared to Runge Kutta and Adams Bashforth and Adams Moulton. This method represents a single step of integration at a point. The Implicit Euler technique, however uses the unknown next value of the iteration in the same setup. This results in an equation that must be solved using some other method such as the Newton method for finding zeroes. The Implicit Euler Method has order one. This means that the local truncation error is \\(O(h^{2})\\). The error at a specific time \\(t\\) is \\(O(h)\\).

Runge Kutta, which is favored of engineers, is favored due to its stability and ease of use. Runge Kutta techniques exist for arbitrary order, the Kunge Kutta order 4 is the most widely known and is referred to as the "classical Runge Kutta method".

The Predictor Corrector method used Adams Bashforth for the prediction step then Adams Moulton for the corrector. This is the typical form that predictor-corrector methods take, an explicit method for the predictor step and an implicit method for the corrector step. The Adams–Bashforth methods are explicit methods. The coefficients are \\(a_{s-1}=-1\\) and \\(a_{s-2}=\cdots =a_0=0\\), while the \\(b_j\\) are chosen such that the methods has order s (this determines the methods uniquely). The Adams–Moulton methods are similar to the Adams–Bashforth methods in that they also have \\(a_{s-1}=-1\\) and \\(a_{s-2}=\cdots =a_0=0\\). Again the b coefficients are chosen to obtain the highest order possible. However, the Adams–Moulton methods are implicit methods. By removing the restriction that \\(b_s = 0\\) , an s-step Adams–Moulton method can reach order s+1, while an s-step Adams–Bashforth methods has only order s.

