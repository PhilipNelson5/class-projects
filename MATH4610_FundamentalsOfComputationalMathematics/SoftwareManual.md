---
layout: default
title: MATH 4610 Software Manual
---

<a href="https://philipnelson5.github.io">Home</a>

# Software Manual

**Basic Routines**
* Machine Epsilon - [maceps](./hw1/1-maceps/manual.md)

**Error Routines**
* Absolute Error - [absolute_error](./hw2/1-error/manual_abs.md)
* Relative Error - [relative_error](./hw2/1-error/manual_rel.md)
* Absolute Vector Error - [absolute_error](./hw3/2-vectorError/manual_abs.md)
* Relative Vector Error - [relative_error](./hw3/2-vectorError/manual_rel.md)

**Root Finding**
* Quadratic Equation - [quadratic_equation](./hw1/7-quadraticEquation/manual.md)
* Fixed Point Iteration - [root_finder_fixed_point_iteration](./hw2/4-fixedPointIteration/manual.md)
* Bisection Method - [root_finder_bisection](./hw2/5-bisection/manual.md)
* Newton's Method - [root_finder_newton](./hw2/6-newton/manual.md)
* Secant Method - [root_finder_secant](./hw2/7-secant/manual.md)
* Hybrid Newton's Method - [root_finder_hybrid_newton](./hw2/8-hybridNewton/manual.md)
* Hybrid Secant Method -  [root_finder_secant_method](./hw2/9-hybridSecant/manual.md)

**Derivative Approximation**
* Definition of the Derivative - [deriv_approx](./hw2/2-derivativeApproximation/manual.md)

**Norms**
* l_p Vector Norms - [l_pNorm](./hw3/1-vectorNorms/manual_l_pNorms.md)
* l Infinity Vector Norm - [l_inf](./hw3/1-vectorNorms/manual_l_inf.md)
* One Matrix Norm - [one_norm](./hw3/4-matrixNorms/manual_one_norm.md)
* Infinity Matrix Norm - [inf_norm](./hw3/4-matrixNorms/manual_inf_norm.md)
* Frobenius Matrix Norm - [frobenius_norm](./hw3/4-matrixNorms/manual_frobenius_norm.md)

**Vector Operations**
* Vector Addition and Subtraction - [Addition / Subtraction](./hw3/3-vectorOperations/manual_vector_addition_subtraction.md)
* Vector Scalar Multiplication - [Vector Scalar Multiplication](./hw3/3-vectorOperations/manual_vector_scalar_multiplication.md)
* Inner Product [inner_product](./hw3/3-vectorOperations/manual_vector_inner_product.md)
* Cross Product [cross_product](./hw3/3-vectorOperations/manual_vector_cross_product.md)
* Outer Product [outer_product](./hw3/8-vectorAdditionalOperations/manual_vector_outer_product.md)
* Orthogonal Basis [orthogonal_basis](./hw3/9-orthogonalBasis/manual_orthogonal_basis.md)

**Matrix Operations**
* Matrix Addition and Subtraction -  [Addition / Subtraction](./hw3/5-matrixOperations/manual_matrix_add_subtract.md)
* Transpose - [Matrix Transpose](./hw3/5-matrixOperations/manual_matrix_transpose.md)
* Trace - [Matrix Trace](./hw3/5-matrixOperations/manual_matrix_trace.md)
* Scalar Matrix Multiplication - [Scalar Matrix Multiplication](./hw3/5-matrixOperations/manual_matrix_scalar_multiplication.md)
* Matrix Vector Multiplication - [Matrix Vector Multiplication](./hw3/5-matrixOperations/manual_matrix_vector_multiplication.md)
* Matrix Matrix Multiplication - [Matrix Matrix Multiplication](./hw3/5-matrixOperations/manual_matrix_matrix_multiplication.md)
* Kronecker Product - [kronecker_product](./hw3/6-matrixAdditionalOperations/manual_kronecker_product.md)
* Matrix Determinant - [determinant](./hw3/6-matrixAdditionalOperations/manual_determinant.md)
* Matrix Outer Product - [a. Matrix Outer Product](./hw3/6-matrixAdditionalOperations/manual_matrix_outer_product.md)

**Generators**
* Matrix Generator[generate_square_symmetric_diagonally_dominant_matrix](./hw5/2-MatrixGenerator/manual_generate_matrix)
* Right Side Generator[generate_right_side](./hw5/2-MatrixGenerator/manual_generate_right_side.md)

**Linear Solvers**
* Solve By Gaussian Elimination - [solve_linear_system_gaussian_elimination](./hw4/5-SolveSystemGaussianElimination/manual_solve_linear_system_gaussian_elimination.md)
  * Gaussian Elimination - [gaussian_elimination](./hw4/1-GaussianElimination/manual_gaussian_elimination.md)
  * Back Substitution - [back_substitution](./hw4/4-BackSubstitution/manual_back_sub.md)
* Solve By LU Factorization [solve_linear_system_lu](./hw4/7-SolveSystemLUFactorization/manual_solve_lu_factorization.md)
  * LU Factorization - [LU_factorization](./hw4/6-LUFactorization/manual_LU_factorization.md)
  * Forward Substitution - [forward_substitution](./hw4/3-ForwardSubstitution/manual_forward_sub.md)
  * Back Substitution - [back_substitution](./hw4/4-BackSubstitution/manual_back_sub.md)
* Solve By System Cholesky Factorization - [solve_linear_system_cholesky](./hw4/10-SolveSystemCholeskyFactorization/manual_solve_cholesky.md)
  * Cholesky Factorization - [cholesky_factorization](./hw4/8-CholeskyFactorization/manual_cholesky_factorization.md)
* Jacobi Iteration - [jacobi_iteration](./hw5/3-JacobiIteration/manual_jacobi_iteration.md)
* Gauss Seidel - [gauss_seidel](./hw5/4-GaussSeidel/manual_gauss_sidel.md)
* Conjugate Gradient / Steepest Descent - [conjugate_gradient](./hw5/5-SteepestDescent/manual_conjugate_gradient.md)
* Parallel Jacobi Iteration - [parallel_jacobi_iteration](./hw5/7-ParallelJacobiIteration/manual_parallel_jacobi_iteration.md)
* Parallel Gauss Seidel - [parallel_gauss_seidel](./hw5/8-ParallelGaussSeidel/manual_parallel_gauss_sidel.md)
* Parallel Conjugate Gradient - [parallel_conjugate_gradient](./hw5/9-ParallelConjugateGradient/manual_parallel_conjugate_gradient.md)

**Eigenvalue Methods**
* Power Method - [power_method](./hw6/1-PowerMethod/manual_power_iteration.md)
* Inverse Power Method - [inverse_power_iteration](./hw6/2-InversePowerMethod/manual_inverse_power_iteration.md)
* 2 Condition Number Estimate - [condition_2_estimate](./hw6/3-Condition2Estimate/manual_condition_estimate.md)
* Parallel Power Method - [parallel_power_iteration](./hw6/4-ParallelPowerMethod/manual_parallel_power_iteration.md)
* Parallel Inverse Power Method - [parallel_inverse_power_iteration](./hw6/5-ParallelInversePowerMethod/manual_parallel_inverse_power_iteration.md)
