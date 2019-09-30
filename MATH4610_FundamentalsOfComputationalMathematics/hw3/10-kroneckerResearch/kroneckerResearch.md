---
title: Kronecker Product
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Machine Precision in Practice

**Author:** Philip Nelson

The Kronecker product is an operation on two matrices of arbitrary size resulting in a block matrix. It is a generalization of the outer product from vectors to matrices, and gives the matrix of the tensor product with respect to a standard choice of basis [1].

The Kronecker Product is used in solving Sylvester and Lupunov equations which is a system of the form \\(AX + XB = C\\) where \\(A\in R^{nxn},B\in R^{mxm},C\in R^{nxm}\\). These systems arise naturally in stability theory [2].

Another application of the Kronecker product arises in defining the search directions for primal-dual interior-point methods in semidefinite programming [3].

1. [Kronecker Product](https://www.wikiwand.com/en/Kronecker_product)

2. [Kronecker Products](https://archive.siam.org/books/textbooks/OT91sample.pdf)

3. [On the Kronecker Product](https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf)

**Last Modified:** October 2018
