---
layout: default
title: MATH 5620 Software Manual
---

<a href="https://philipnelson5.github.io">Home</a>

# Table of Contents

_(Homework problems below)_

### Basic Routines
- [Machine Epsilon](./machineEpsilon/manual)
- [Absolute and Relative Error](./error/manual)
- [Logistic Differential Equation](./logistic/manual)
- [Second Order Linear DE with Constant Coefficients](./secondOrderLinear/manual)

### Norms
- [Vector P Norm](./matrix/manual_vector_pnorm)
- [Vector Infinity Norm](./matrix/manual_vector_infinity_norm)
- [Matrix One Norm](./matrix/manual_matrix_one_norm)
- [Matrix Infinity Norm](./matrix/manual_matrix_infinity_norm)

### Linear Solvers
- [Thomas Algorithm](./matrix/manual_thomas_algorithm)
- [Jacobi Iteration](./matrix/manual_jacobi_iteration)
- [Conjugate Gradient](./conjugateGradient/manual_conjugate_gradient)
- [Linear Solver by LU Factorization](./matrix/manual_linear_solve_lu)
  - [Lu Factorization](./matrix./manual_lu_factorization)
  - [Forward Substitution](./matrix./manual_forward_sub)
  - [Back Substitution](./matrix./manual_back_sub)

### Eigenvalue Methods
- [Power Method Iteration for Largest Eigenvalue](./matrix/manual_power_iteration)
- [Inverse Power Method for Smallest Eigenvalue](./matrix/manual_inverse_power_iteration)
- [Power Method Iteration for Solving 2nd Order FD Elliptic ODE](./matrix/example_power_iteration_elliptic_ode)
- [Inverse Power Method for Solving 2nd Order FD Elliptic ODE](./matrix/example_inverse_power_iteration_elliptic_ode)

### Elliptic Problems
- [Finite Difference Coefficients](./finiteDiffMethods/manual_finite_diff_coeff)
- [Initialize Elliptic ODE](./finiteDiffMethods/manual_init_elliptic_ode)
- [Solve Elliptic ODE](./finiteDiffMethods/manual_solve_elliptic_ode)
- [Solve Laplace Equation with 5-point Stencil](./matrix/manual_solve_five_point_stencil)
  - [Generate 5-point Stencil](./matrix/manual_gen_five_point_stencil)
  - [Generate Mesh](./matrix/manual_gen_mesh)
  - [Initialize B for Mesh](./matrix/manual_init_b)

### First Order IVPs
- [Explicit Euler](./explicitEuler/manual_explicit_euler)
- [Implicit Euler](./implicitEuler/manual_implicit_euler)
  - [Newton's Method](./newtonsMethod/manual_newtons_method)
- [Runge Kutta order 2](./rungeKuttaOrder2/manual_runge_kutta_order2)
- [Runge Kutta order 4](./rungeKuttaOrder4/manual_runge_kutta_order4)
- [Predicticor Corrector - Adams Bashforth / Adams Moulton](./predictorCorrector/manual_predictor_corrector)

### Parabolic Problems
- [Upwinding](./upwinding/manual_upwinding)
- [Lax-Wendorff Method](./laxWendroff/manual_lax_wendroff)
- [Warming and Beam Method](./warmingAndBeam/manual_warming_and_beam)

---

### Homework 1
*due: 25 January 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 2-3.**  | [Machine Epsilon](./machineEpsilon/manual)|
| **Problem 4.**    | [Absolute and Relative Error](./error/manual)|
| **Problem 6.**    | [Logistic Differential Equation](./logistic/manual)|
| **Problem 7.**    | [Second Order Linear DE with Constant Coefficients](./secondOrderLinear/manual)|


### Homework 2
*due: 8 February 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 1-2.**  | [Finite Difference Coefficients](./finiteDiffMethods/manual_finite_diff_coeff)|
| **Problem 3.**    | [Initialize Elliptic ODE](./finiteDiffMethods/manual_init_elliptic_ode)|
| **Problem 4.**    | [Thomas Algorithm](./matrix/manual_thomas_algorithm)|
| **Problem 5.**    | [Linear Solver by LU Factorization](./matrix/manual_linear_solve_lu)|
|                   | [Lu Factorization](./matrix/manual_lu_factorize)|
|                   | [Forward Substitution](./matrix/manual_forward_sub)|
|                   | [Back Substitution](./matrix/manual_back_sub)|
| **Problem 6.**    | [Jacobi Iteration](./matrix/manual_jacobi_iteration)|
| **Problem 8.**    | [Solve Elliptic ODE](./finiteDiffMethods/manual_solve_elliptic_ode)|
|                   | [Vector P Norm](./matrix/manual_vector_pnorm)|
|                   | [Vector Infinity Norm](./matrix/manual_vector_infinity_norm)|

### Homework 3
*due: 1 March 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 1.**    | [Matrix One Norm](./matrix/manual_matrix_one_norm)|
|                   | [Matrix Infinity Norm](./matrix/manual_matrix_infinity_norm)|
| **Problem 2.**    | [Power Method Iteration for Largest Eigenvalue](./matrix/manual_power_iteration)|
|                   | [Inverse Power Method for Smallest Eigenvalue](./matrix/manual_inverse_power_iteration)|
| **Problem 3.**    | [Power Method Iteration for Solving 2nd Order FD Elliptic ODE](./matrix/example_power_iteration_elliptic_ode)|
| **Problem 4.**    | [Inverse Power Method for Solving 2nd Order FD Elliptic ODE](./matrix/example_inverse_power_iteration_elliptic_ode)|
| **Problem 5.**    | [Solve Laplace Equation with 5-point Stencil](./matrix/manual_solve_five_point_stencil)|
|                   | [Generate 5-point Stencil](./matrix/manual_gen_five_point_stencil)|
|                   | [Generate Mesh](./matrix/manual_gen_mesh)|
|                   | [Initialize B for Mesh](./matrix/manual_init_b)|
| **Problem 6.**    | [Solve Laplace Equation with 9-point Stencil](./matrix/manual_solve_nine_point_stencil)|
|                   | [Generate 9-point Stencil](./matrix/manual_gen_nine_point_stencil)|

### Homework 4
*due: 22 March 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 1.**    | [Gauss-Seidel](./gaussSidel/manual_gauss_sidel)|
| **Problem 2.**    | [Conjugate Gradient](./conjugateGradient/manual_conjugate_gradient)|
| **Problem 3.**    | [Application of Conjugate Gradient](./testConjugateGradientFivePoint/manual_solve_five_point_stencil_test)|
| **Problem 4.**    | [Explicit Euler](./explicitEuler/manual_explicit_euler)|

### Homework 5
*due: 3 April 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 1.**    | [First Order IVP Test](./5.1IVP/IVP_test)|
|                   | [Logistic Model Test](./logistic2/manual)|
| **Problem 2.**    | [IVP via Explicit Euler Test](./explicitEulerTest/manual_explicit_euler_test)|
| **Problem 3.**    | [Implicit Euler](./implicitEuler/manual_implicit_euler)|
|                   | [Newton's Method](./newtonsMethod/manual_newtons_method)|
| **Problem 4.**    | [Runge Kutta order 2](./rungeKuttaOrder2/manual_runge_kutta_order2)|
|                   | [Runge Kutta order 4](./rungeKuttaOrder4/manual_runge_kutta_order4)|
| **Problem 5.**    | [Adam's Bashford](./predictorCorrector/manual_predictor_corrector)|
| **Problem 6.**    | [Summary of Iterative Methods](./SumaryOfIterativeMethods)|

### Homework 6
*due: 24 April 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Write up**      | [Experiments 7.1, 7.2, 7.4](./hw6_experiments.md)|

### Homework 7
*due: 5 May 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 1.**    | [Heat Equation - Explicit Euler](./heatEquations/manual_heat_equation_explicit_euler)|
| **Problem 2.**    | [Heat Equation - Implicit Euler](./heatEquations/manual_heat_equation_implicit_euler)|
| **Problem 4.**    | [Heat Equation - Predictor Corrector](./heatEquations/manual_heat_equation_predictor_corrector)|
| **Problem 5.**    | [Heat Equation - Runge Kutta Order 4](./heatEquations/manual_heat_equation_runge_kutta)|
{% comment %}
| **Problem 3.**    | [Changing Time Time Step](./heatEquations/manual_heat_equation)|
{% endcomment %}

### Homework 8
*due: 5 May 2018*

| Problem           | Software Manual|
| :-----------------|:---------------|
| **Problem 1.1**   | [Upwinding](./upwinding/manual_upwinding)|
| **Problem 1.2**   | [Lax-Wendorff Method](./laxWendroff/manual_lax_wendroff)|
| **Problem 1.3**   | [Warming and Beam Method](./warmingAndBeam/manual_warming_and_beam)|
| **Problem 2.1**   | [vonNeuman Stability Analysis Lax Wendroff](./stabilityAnalysis/laxWendroff)|
| **Problem 2.2**   | [vonNeuman Stability Analysis Warming and Beam](./stabilityAnalysis/warmingAndBeam)|
