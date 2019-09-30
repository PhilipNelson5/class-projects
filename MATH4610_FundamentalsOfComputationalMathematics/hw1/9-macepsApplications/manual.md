---
title: Machine Precision in Practice
layout: default
---
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Machine Precision in Practice

**Author:** Philip Nelson

Machine Learning is an area of computer science and that is influenced by machine precision. The training of deep neural networks can be constrained by the available computational resources [^fn]. In a paper published by Suyog Gupta et al entitled _Deep Learning with Limited Numerical Precision_, Gupta studies the effect of low-precision, fixed-point computations and how different rounding schemes affect training and classification accuracy of deep neural networks. Gupta's finds that using 16-bit wide fixed-point representation while employing stochastic rounding incurs little to no degradation in classification accuracy. By substituting floating-point numeric representation and arithmetic for fixed-point, a deep neural network was trained using only 16-bit fixed point and performed "nearly identical" to a 32-bit floating point network. Gupta is additionally able to create an energy efficient architecture for matrix multiplication that relies on stochastic rounding.

**Last Modified:** September 2018

[^fn]: Suyog Gupta, Ankur Agrawal, and Kailash Gopalakrishnan. "Deep Learning with Limited Numerical Precision." arxiv.org. February 9, 2015. Accessed September 3, 2018. https://arxiv.org/pdf/1502.02551.pdf.
