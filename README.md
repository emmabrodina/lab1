# Lab1
Solving numerical analysis problems for elliptic equations using ADI, block tridiagonal solver, and relaxation method

# Numerical Methods for 2D Elliptic Equations

This repository contains implementations of numerical methods for solving two-dimensional elliptic equations. The main methods included are:

1. **ADI (Alternating Direction Implicit) Method**: Used to solve 2D problems by breaking them into sequential 1D subproblems.
2. **Block Thomas Algorithm**: A generalized tridiagonal solver for block matrices, suitable for 2D discretized problems.
3. **SOR (Successive Over-Relaxation) Method**: An iterative method for faster convergence when solving large grid elliptic equations.

## Folder Structure

The repository consists of the following files:

1. **adi_method** - Implementation of the ADI method.
2. **block_thomas_method** - Implementation of the Block Thomas Algorithm.
3. **sor_method** - Implementation of the SOR method.
4. **task_9_solved** - Solution for Task 9 from Lab 1, corresponding to the group number "2" for Task 1.
5. **task_17_solved** - Solution for Task 17 from Lab 1, corresponding to the group number "2" for Task 2.

All files include comments to facilitate understanding of the code. Note that boundary conditions are specified in the code along with internal conditions, which is relevant for the second task.

## Task-Solving Sequence

1. **Thomas Algorithm and ADI Method Implementation**:
   - The Thomas Algorithm was implemented first to solve tridiagonal systems, as it is used in the ADI method.
   - The ADI solver was then created to solve 2D problems by splitting them into sequential 1D subproblems using the Thomas Algorithm. The choice of this order is based on:
     1. The ADI method simplifies the 2D problem by reducing it to a series of 1D problems.
     2. It is easier to implement than the Block Thomas method because each 1D subproblem can be solved using the standard Thomas Algorithm.

2. **Block Thomas Method**:
   - Implemented after ADI, as it generalizes the Thomas Algorithm for block tridiagonal systems where each matrix element is a submatrix (block). This structure is typical for 2D discretized problems.

3. **SOR Method**:
   - The SOR (Successive Over-Relaxation) method was implemented last to achieve faster convergence, especially for solving large-grid elliptic equations. The adjustable relaxation parameter \( \omega \) allows for tuning to improve the rate of convergence.

4. **Solutions for Tasks**:
   - The solutions are presented with LaTeX-formatted equations, including coefficients and boundary/internal conditions, followed by visualization and conclusions.

## How to Use

- Clone the repository and navigate to the relevant files for the specific method or task.
- Each method can be executed independently to solve 2D elliptic equation problems.
- The tasks demonstrate practical applications and provide detailed comments to help understand the process.

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization)
