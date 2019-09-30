---
title: Warming and Beam Analysis
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Warming and Beam Stability Analysis

**Author:** Philip Nelson
(see also p.213 of text)

\\(g(\xi) = 1 - \frac{v}{2}(3 - 4e^{-i\xi h} + e^{-i\xi 2h}) + \frac{v^2}{2}(1 - 2e^{-i\xi h} + e^{-i\xi 2h})\\)
\\(\implies e^{i\xi h} g(\xi) = e^{i\xi h} - \frac{v}{2}(3e^{i\xi h} - 4 + e^{-i\xi h}) + \frac{v^2}{2}(e^{i\xi h} - 2 + e^{-i\xi h})\\)
\\(= e^{i\xi h} - \frac{v}{2}(3e^{i\xi h} - 4 + e^{-i\xi h}) - v^2 + v^2\cos(\xi h)\\)
\\(= e^{i\xi h} - \frac{v}{2}(2e^{i\xi h} - 4) + v^2(\cos(\xi h) - 1)\\)
\\(= e^{i\xi h}(1 - v) - 2v + v\cos(\xi h) + v^2(\cos(\xi h) - 1)\\)
\\(= e^{i\xi h}(1 - v) - v(2 - \cos(\xi h)) + v^2(\cos(\xi h) - 1)\\)
\\(= (\cos(\xi h) + i\sin(\xi h))(1 - v) - v(2 - \cos(\xi h)) + v^2(\cos(\xi h) - 1)\\)
\\(\implies |g(\xi)|^2 = sin^2(\xi h) + \Big( \cos(\xi h)(1 - v) - v(2 - \cos(\xi h)) + v^2(\cos(\xi h) - 1) \Big)^2\\)
\\(\implies \text{stable for } 0 \leq v \leq 2\\)

**Last Modification date:** 5 May 2018
