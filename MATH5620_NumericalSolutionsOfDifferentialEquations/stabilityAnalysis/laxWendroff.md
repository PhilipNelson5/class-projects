---
title: laxWendroff Analysis
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Lax Wendroff Stability Analysis

**Author:** Philip Nelson
(see also p.213 of text)

\\(g(\xi) = 1 - \frac{1}{2}v(e^{i\xi h} - e^{-i\xi h}) + \frac{1}{2}v^2(e^{i\xi h} - 2 + e^{-i\xi h})\\)
\\(= 1 - iv\sin(\xi h) + v^2(\cos(\xi h) - 1)\\)
\\(= 1 - iv \Big( 2\sin(\frac{\xi h}{2})cos(\frac{\xi h}{2} \Big) + v^2 \Big( 2sin^2(\frac{\xi h}{2}) \Big)\\)
\\( \implies |g(\xi)|^2 = \Big( 1 - 2sin^2(\frac{\xi h}{2}) \Big)^2 + 4\sin^2(\frac{\xi h}{2})cos^2(\frac{\xi h}{2})\\)
\\(= 1 - 4v^2(1 - v^2)sin^4(\frac{\xi h}{2})\\)
\\( \implies \text{stable for } |v| \leq 1\\)

**Last Modification date:** 5 May 2018
