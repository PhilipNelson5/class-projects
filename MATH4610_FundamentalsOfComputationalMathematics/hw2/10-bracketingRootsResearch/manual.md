---
title: Machine Precision in Practice
layout: default
---
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Machine Precision in Practice

**Author:** Philip Nelson

Bisection is a simple bracketing method to find the roots of a nonlinear equation. It starts with two initial values \\(a\\) and \\(b\\) where \\(f(a)\cdot f(b) < 0\\) which means that by the intermediate value theorem a zero must exist in between \\(a\\) and \\(b\\). Then the interval is split in half and the new interval is evaluated [1].

The Bisection method is often called a bracket method, because every interval brackets the root. However, the Newton method and the secant method are not bracket methods in that sense, because there is no guarantee that the two successive approximations will bracket the root [2].

1. [A Review of Bracketing Methods for Finding Zeros of Nonlinear Functions](http://www.m-hikari.com/ams/ams-2018/ams-1-4-2018/p/intepAMS1-4-2018.pdf)

2. [Numerical Methods for the Root Finding Problem](http://www.math.niu.edu/~dattab/MATH435.2013/ROOT_FINDING.pdf)

3. [A Global root bracketing method with adaptive mesh refinement](https://arxiv.org/pdf/1501.05298.pdf)

4. [Applications of Root Finding](https://www.reed.edu/physics/courses/P200.L.S11/Physics200Lab/files/Bisection.pdf)

**Last Modified:** October 2018
