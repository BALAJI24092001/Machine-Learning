<span style="font-family: 'Times New Roman'; font-size: 1.15em;">

<span align = 'center'>

# LINEAR ALGEBRA

   </span>

**Matrices:**

- Elements in a rectangular array with numbers in rows and columns.
- Order of a matrix is given by $m \times n$ where $m$ is the number of rows and $n$ is the number of columns.
- Number of elements in a matrix is $m \times n$.

**Matrix Representation:**

- A matrix $A = [a_{ij}]_{m \times n}$ where $i$ represents the row number and $j$ represents the column number.
- Horizontal elements are called rows.
- Vertical elements are called columns.

## Types of Matrices

1. **Row Matrix:**

   - Having only one row.
   - Represented as $A = [a_{1j}]$ where $1 \leq j \leq n$.

2. **Column Matrix:**

   - Having only one column.
   - Represented as $A = [a_{i1}]$.

3. **Horizontal Matrix:**

   - Number of columns (n) > number of rows (m).

4. **Vertical Matrix:**

   - Number of columns (n) < number of rows (m).

5. **Zero Matrix:**

   - Matrix of any order with all entries as 0.

6. **Triangular Matrices**:

   - **Upper Triangular**: $a_{ij} = 0$ for $i > j$.
     - Example:
       $$
       \begin{bmatrix}
       a & b & c \\
       0 & d & e \\
       0 & 0 & f
       \end{bmatrix}
       $$
   - **Lower Triangular**: $a_{ij} = 0$ for $i < j$.

     - Example:

       $$
       \begin{bmatrix}
       a & 0 & 0 \\
       b & c & 0 \\
       d & e & f
       \end{bmatrix}
       $$

     **Number of Minimum Zeroes in Triangular Matrices**:<br>

   - Formula: $\frac{n^2 - n}{2}$.

     **Product of Diagonal Elements**:<br>

   - For triangular matrices: $| \text{triangular} | = \text{product of diagonal elements} = | \text{diagonal matrix} |$.
     <br>

7. **Diagonal Matrices**:

   - **Diagonal**: $a_{ij} = 0$ for $i \neq j$.
     - Example:
       $$
       \begin{bmatrix}
       d_1 & 0 & 0 \\
       0 & d_2 & 0 \\
       0 & 0 & d_3
       \end{bmatrix}
       $$
   - **Scalar**: All diagonal elements are equal.
     - Example:
       $$
       \begin{bmatrix}
       a & 0 & 0 \\
       0 & a & 0 \\
       0 & 0 & a
       \end{bmatrix}
       $$
   - **Unit Matrix**: All diagonal elements are 1.
     - Example:
       $$
       \begin{bmatrix}
       1 & 0 & 0 \\
       0 & 1 & 0 \\
       0 & 0 & 1
       \end{bmatrix}
       $$

## Properties

1. **Commutative Property of Addition**:

   - $A + B = B + A$

2. **Associative Property of Addition**:

   - $(A + B) + C = A + (B + C)$

3. **Associative Property of Multiplication**:

   - $(AB)C = A(BC)$

4. **Distributive Property**:

   - $A(B + C) = AB + AC$
   - $(A + B)C = AC + BC$

5. **Identity Matrix**:

   - Multiplying a matrix by the identity matrix $I$ does not change the matrix: $AI = IA = A$.

### Trace of a Matrix

Let $A$ be a square matrix, then the trace of matrix a; denoted by $\text{TR}(\mathbf{A})$ is sum of all the diagonal elements.
i.e.. $\sum_{i=1}^{n}A_{i,i}\quad$ where $n$ is dimension of the matrix.

$$
\begin{aligned}
\text{Tr}(\mathbf{A}) &= \sum_i A_{ii} \\
\text{Tr}(\mathbf{A}) &= \sum_i \lambda_i, \quad \lambda_i = \text{eig}(\mathbf{A}) \\
\text{Tr}(\mathbf{A}) &= \text{Tr}(\mathbf{A}^T) \\
\text{Tr}(\mathbf{AB}) &= \text{Tr}(\mathbf{BA}) \\
\text{Tr}(\mathbf{A} + \mathbf{B}) &= \text{Tr}(\mathbf{A}) + \text{Tr}(\mathbf{B}) \\
\text{Tr}(\mathbf{ABC}) &= \text{Tr}(\mathbf{BCA}) = \text{Tr}(\mathbf{CAB}) \\
\mathbf{a}^T \mathbf{a} &= \text{Tr}(\mathbf{aa}^T)
\end{aligned}
$$

### Matrix Multiplication

Matrix multiplication is defined as the dot product of rows and columns between two matrices. The number of columns in the first matrix must equal the number of rows in the second matrix.

If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, their product $AB$ is an $m \times p$ matrix:

$$
C = AB, \quad \text{where} \ C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

For example, for $A$ (2x3) and $B$ (3x2):

$$
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{pmatrix}, \quad
B = \begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
b_{31} & b_{32}
\end{pmatrix}
$$

Their product is:

$$
AB = \begin{pmatrix}
a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} & a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32} \\
a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31} & a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32}
\end{pmatrix}
$$

### Matrix Addition

Two matrices of the same dimensions can be added element-wise:

If $A$ and $B$ are both $m \times n$ matrices, then their sum $C$ is:

$$
C = A + B, \quad \text{where} \ C_{ij} = A_{ij} + B_{ij}
$$

For example:

$$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}, \quad
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}
$$

Their sum is:

$$
A + B = \begin{pmatrix}
6 & 8 \\
10 & 12
\end{pmatrix}
$$

let $C = A + B$ then $C^2$ is

$$
C^2 \neq A^2 + B^2 + 2AB \\
$$

$$
C^2 = (A+B)(A+B) \\ \quad \quad\quad = A^2 + AB + BA + B^2
$$

### Hadamard Product

The Hadamard product is the element-wise multiplication of two matrices of the same dimensions. If $A$ and $B$ are both $m \times n$ matrices, their Hadamard product $C$ is:

$$
C = A \circ B, \quad \text{where} \ C_{ij} = A_{ij} \times B_{ij}
$$

For example:

$$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}, \quad
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}
$$

Their Hadamard product is:

$$
A \circ B = \begin{pmatrix}
1 \times 5 & 2 \times 6 \\
3 \times 7 & 4 \times 8
\end{pmatrix}
= \begin{pmatrix}
5 & 12 \\
21 & 32
\end{pmatrix}
$$

### Determinant

The determinant is a scalar value that can be computed from a square matrix. It provides important information about the matrix, such as whether it is invertible or how it scales volumes. Let's delve into the details:

**Determinant of a 2x2 Matrix**

For a matrix

$$
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

the determinant, denoted as $\text{det}(A)$ or $|A|$, is calculated as:

$$
\text{det}(A) = ad - bc
$$

**Determinant of a 3x3 Matrix**

For a matrix

$$
A = \begin{pmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{pmatrix}
$$

the determinant is given by:

$$
\text{det}(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
$$

**Determinant of an $n \times n$ Matrix**

For larger square matrices, the determinant can be calculated using various methods, such as cofactor expansion or row reduction.

**Cofactor Expansion**<br>

To compute the determinant using cofactor expansion along the first row, for a matrix

$$
A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{pmatrix}
$$

$$
\text{det}(A) = \sum_{j=1}^{n} (-1)^{1+j} a_{1j} \text{det}(A_{1j})
$$

where $A_{1j}$ is the $(n-1) \times (n-1)$ submatrix obtained by removing the first row and the $j$-th column from $A$.

**Properties of Determinants**

1. **Row Operations**:

   - Swapping two rows of a matrix multiplies its determinant by -1.
   - Multiplying a row by a scalar multiplies the determinant by that scalar.
   - Adding a multiple of one row to another row does not change the determinant.

2. **Invertibility**: A matrix is invertible (non-singular) if and only if its determinant is non-zero.

3. **Product**: The determinant of the product of two matrices is the product of their determinants:

   $$
   \text{det}(AB) = \text{det}(A) \cdot \text{det}(B)
   $$

4. **Transpose**: The determinant of a matrix is equal to the determinant of its transpose:

   $$
   \text{det}(A) = \text{det}(A^T)
   $$

5. **Triangular Matrices**: The determinant of a triangular matrix (upper or lower) is the product of its diagonal elements.

Determinants are widely used in solving systems of linear equations, analyzing linear transformations, and in various fields such as physics and engineering. If you have any specific matrix in mind or further questions, feel free to ask!

### Inverse of a Matrix

#### Gaussian Elimination Method

1. **Set Up the Augmented Matrix**:

   - Start with the matrix $A$ that you want to invert, and augment it with the identity matrix of the same size. This creates a $[A | I]$ matrix.

2. **Apply Row Operations**:

   - Perform a series of row operations to transform $A$ into the identity matrix $I$. The row operations can include:
     - Swapping two rows
     - Multiplying a row by a non-zero scalar
     - Adding or subtracting a multiple of one row to/from another row
   - Apply the same row operations to the identity matrix on the right side of the augmented matrix. The goal is to make $A$ look like $I$, and the identity matrix will transform into $A^{-1}$.

3. **Final Augmented Matrix**:
   - Once the left side of the augmented matrix is the identity matrix, the right side will be the inverse of $A$.

##### Example

To determine the inverse of

$$
A = \begin{bmatrix}
1 & 0 & 2 & 0 \\
1 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

we write down the augmented matrix

$$
\begin{bmatrix}
1 & 0 & 2 & 0 & | & 1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & | & 0 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 & | & 0 & 0 & 1 & 0 \\
1 & 1 & 1 & 1 & | & 0 & 0 & 0 & 1
\end{bmatrix}
$$

and use Gaussian elimination to bring it into reduced row-echelon form

$$
\begin{bmatrix}
1 & 0 & 0 & 0 & | &-1 & 2 & -2 & 2 \\
0 & 1 & 0 & 0 & | & 1 & -1 & 2 & -2 \\
0 & 0 & 1 & 0 & | & 1 & -1 & 1 & -1 \\
0 & 0 & 0 & 1 & | & -1 & 0 & -1 & 2
\end{bmatrix}
$$

such that the desired inverse is given as its right-hand side:

$$
A^{-1} = \begin{bmatrix}
-1 & 2 & -2 & 2 \\
1 & -1 & 2 & -2 \\
1 & -1 & 1 & -1 \\
-1 & 0 & -1 & 2
\end{bmatrix}
$$

We can verify that this is indeed the inverse by performing the multiplication $AA^{-1}$ and observing that we recover $I_4$.

#### Cofactor Matrix Method

1. **Find the Cofactor Matrix**:

   - For each element $a_{ij}$ of the matrix $A$, compute the cofactor $C_{ij}$.
   - The cofactor $C_{ij}$ is calculated as:
     $$
     C_{ij} = (-1)^{i+j} \text{det}(A_{ij})
     $$
     where $A_{ij}$ is the $(n-1) \times (n-1)$ submatrix obtained by removing the $i$-th row and $j$-th column from $A$.

2. **Form the Adjugate (or Adjoint) Matrix**:

   - The adjugate matrix is the transpose of the cofactor matrix. If the cofactor matrix is $C$, then the adjugate matrix $\text{adj}(A)$ is:
     $$
     \text{adj}(A) = C^T
     $$

3. **Calculate the Determinant of the Original Matrix**:

   - The determinant of matrix $A$, denoted as $\text{det}(A)$, must be non-zero for the inverse to exist.

4. **Compute the Inverse Matrix**:
   - The inverse of matrix $A$, denoted as $A^{-1}$, is given by:
     $$
     A^{-1} = \frac{1}{\text{det}(A)} \text{adj}(A)
     $$

##### Example

Given the matrix $A$:

$$
A = \begin{pmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
5 & 6 & 0
\end{pmatrix}
$$

Let's follow the detailed steps to find its inverse:

1. **Calculate the Determinant of $ A $**:

   $$
   \text{det}(A) = 1 \cdot (1 \cdot 0 - 4 \cdot 6) - 2 \cdot (0 \cdot 0 - 4 \cdot 5) + 3 \cdot (0 \cdot 6 - 1 \cdot 5)
   $$

   $$
   \text{det}(A) = 1 \cdot (0 - 24) - 2 \cdot (0 - 20) + 3 \cdot (0 - 5)
   $$

   $$
   \text{det}(A) = -24 + 40 - 15
   $$

   $$
   \text{det}(A) = 1
   $$

2. **Find the Cofactor Matrix**:

- For each element $a_{ij}$ of $A$, compute the cofactor $C_{ij}$:
  $$
  \text{Cofactors}(A) = \begin{pmatrix}
  \text{det}\begin{pmatrix} 1 & 4 \\ 6 & 0 \end{pmatrix} & -\text{det}\begin{pmatrix} 0 & 4 \\ 5 & 0 \end{pmatrix} & \text{det}\begin{pmatrix} 0 & 1 \\ 5 & 6 \end{pmatrix} \\
  -\text{det}\begin{pmatrix} 2 & 3 \\ 6 & 0 \end{pmatrix} & \text{det}\begin{pmatrix} 1 & 3 \\ 5 & 0 \end{pmatrix} & -\text{det}\begin{pmatrix} 1 & 2 \\ 5 & 6 \end{pmatrix} \\
  \text{det}\begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix} & -\text{det}\begin{pmatrix} 1 & 3 \\ 0 & 4 \end{pmatrix} & \text{det}\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}
  \end{pmatrix}
  $$

$$
\text{Cofactors}(A) = \begin{pmatrix}
-24 & 20 & -5 \\
18 & -15 & 7 \\
5 & -4 & 1
\end{pmatrix}
$$

3. **Form the Adjugate (Transpose of the Cofactor Matrix)**:

   $$
   \text{Adj}(A) = \begin{pmatrix}
   -24 & 18 & 5 \\
   20 & -15 & -4 \\
   -5 & 7 & 1
   \end{pmatrix}
   $$

4. **Compute the Inverse Matrix**:
   $$
   A^{-1} = \frac{1}{\text{det}(A)} \text{Adj}(A)
   $$
   Since $\text{det}(A) = 1$:
   $$
   A^{-1} = \begin{pmatrix}
   -24 & 18 & 5 \\
   20 & -15 & -4 \\
   -5 & 7 & 1
   \end{pmatrix}
   $$
   So, the inverse of the matrix $A$ is:
   $$
   A^{-1} = \begin{pmatrix}
   -24 & 18 & 5 \\
   20 & -15 & -4 \\
   -5 & 7 & 1
   \end{pmatrix}
   $$

## Four Fundamental Sub-Spaces

In linear algebra, the four fundamental subspaces provide a deep understanding of linear transformations and the structure of matrices. These subspaces are associated with a given matrix $A$, typically of dimensions $m \times n$ (where $m$ is the number of rows and $n$ is the number of columns). Let's explore each of these subspaces in detail:

### 1. Column Space (C(A) or Col(A))

- **Definition**: The column space of a matrix $A$ is the set of all possible linear combinations of its column vectors.
- **Dimensionality**: The dimension of the column space is called the rank of the matrix, denoted as $\text{rank}(A)$. The rank represents the number of linearly independent columns in $A$.
- **Interpretation**: The column space represents all the possible vectors that can be produced by the linear transformation represented by $A$.
- **Mathematical Formulation**:
  $$
  C(A) = \{ \mathbf{y} \in \mathbb{R}^m \mid \mathbf{y} = A\mathbf{x} \text{ for some } \mathbf{x} \in \mathbb{R}^n \}
  $$

### 2. Null Space (N(A) or Null(A))

- **Definition**: The null space of a matrix $A$ is the set of all vectors $\mathbf{x}$ that satisfy $A\mathbf{x} = \mathbf{0}$.
- **Dimensionality**: The dimension of the null space is called the nullity of the matrix, denoted as $\text{nullity}(A)$. The nullity is given by $n - \text{rank}(A)$.
- **Interpretation**: The null space represents all the vectors that are mapped to the zero vector by the linear transformation represented by $A$.
- **Mathematical Formulation**:
  $$
  N(A) = \{ \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0} \}
  $$

### 3. Row Space (Row(A))

- **Definition**: The row space of a matrix $A$ is the set of all possible linear combinations of its row vectors.
- **Dimensionality**: The dimension of the row space is equal to the rank of the matrix, $\text{rank}(A)$.
- **Interpretation**: The row space represents all the possible vectors that can be produced by the linear combinations of the row vectors of $A$.
- **Mathematical Formulation**:
  $$
  \text{Row}(A) = \{ \mathbf{y} \in \mathbb{R}^n \mid \mathbf{y} = \mathbf{x}^T A \text{ for some } \mathbf{x} \in \mathbb{R}^m \}
  $$

### 4. Left Null Space ($N(A^T)$)

- **Definition**: The left null space of a matrix $A$ is the null space of its transpose, $A^T$. It is the set of all vectors $\mathbf{y}$ that satisfy $A^T \mathbf{y} = \mathbf{0}$.
- **Dimensionality**: The dimension of the left null space is given by $m - \text{rank}(A)$.
- **Interpretation**: The left null space represents all the vectors that are mapped to the zero vector by the linear transformation represented by $A^T$.
- **Mathematical Formulation**:

  $$
  N(A^T) = \{ \mathbf{y} \in \mathbb{R}^m \mid A^T \mathbf{y} = \mathbf{0} \}
  $$

- **Rank-Nullity Theorem**: The rank-nullity theorem ties the dimensions of the column space and null space together. For an $m \times n$ matrix $A$:
  $$
  \text{rank}(A) + \text{nullity}(A) = n
  $$
- **Orthogonality**: The null space of $A$ and the row space of $A$ are orthogonal complements in $\mathbb{R}^n$. Similarly, the left null space of $A$ and the column space of $A$ are orthogonal complements in $\mathbb{R}^m$.

These four subspaces provide critical insights into the properties and behaviors of linear transformations and matrices. They help us understand solutions to systems of linear equations, the invertibility of matrices, and more.

<span align='center'>

![Four Fundamental Subspaces](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/50ce4d8cddfa06b9c4d84f7e03a7e0e7_Unit_1_WIDE.jpg)

</span>

$$
   \begin{pmatrix}
   x_1 \\
   x_2 \\
   . \\
   . \\
   x_n
   \end{pmatrix} y =
   \begin{pmatrix}
   0 \\
   0 \\
   . \\
   . \\
   0
   \end{pmatrix}
$$

$x_1$ is a row vector belongs to Row space of matrix A, then y is orthogonal to Row Space. i.e., the null space of A is orthogonal to row space and col space is orthogonal to left null space of matrix A respectively.

## System of Linear Equations

A **system of linear equations** is a set of two or more linear equations with the same set of variables. Each equation represents a straight line, and the solution to the system is the point or points where the lines intersect.

A general form of a system of two linear equations with two variables is:

$$
\begin{aligned}
a_1x + b_1y &= c_1 \\
a_2x + b_2y &= c_2
\end{aligned}
$$

Where:

- $a_1, a_2, b_1, b_2, c_1, c_2$ are constants
- $x \; and \; y$ are variables

#### Types of Solutions

1. **Unique Solution**: The lines intersect at a single point.
2. **Infinite Solutions**: The lines are coincident (i.e., they lie on top of each other).
3. **No Solution**: The lines are parallel and never intersect.

**Geometric Interpretation of Systems of Linear Equations**: In a system of linear equations with two variables x1, x2, each linear equation defines a line on the x1x2-plane. Since a solution to a system of linear equations must satisfy all equations simultaneously, the solution set is the intersection of these lines. This intersection set can be a line (if the linar equations describe the same line), a point, or empty (when the lines are parallel)

Similarly, for three variables, each linear equation determines a plane in three-dimensional space. When we intersect these planes, i.e., satisfy all linear equations at the same time, we can obtain a solution set that is a plane, a line, a point or empty (when the planes have no common intersection).

#### Matrix Form of System of Linear Equations

The system of linear equations can be written in matrix form as:

$$
Ax = b
$$

Where:

- $A$ is the matrix of coefficients
- $x$ is the column vector of variables
- $b$ is the column vector of constants

For a system of 2 equations:

$$
\begin{aligned}
a_1x + b_1y &= c_1 \\
a_2x + b_2y &= c_2
\end{aligned}
$$

This can be represented in matrix form as:

$$
\begin{pmatrix}
a_1 & b_1 \\ a_2 & b_2
\end{pmatrix}
\begin{pmatrix}
x \\
y
\end{pmatrix} =
\begin{pmatrix}
c_1 \\
c_2
\end{pmatrix}
$$

The system of linear equations in individual vector multiplication form is:

$$
x
\begin{pmatrix}
a_1 \\
a_2
\end{pmatrix}
+
y \begin{pmatrix}
b_1 \\
b_2
\end{pmatrix} =
\begin{pmatrix}
c_1 \\
c_2
\end{pmatrix}
$$

#### Gaussian Elimination

Key to solving a system of linear equations are elementary transformations that keep the solution set the same, but that transform the equation system into a simpler form:

- Exchange of two equations (rows in the matrix representing the system of equations)
- Multiplication of an equation (row) with a constant λ ∈ R\{0}
- Addition of two equations (rows)

Note: The `augmented matrix` [A| b] compactly represents the system of linear equations Ax = b.

> (Pivots and Staircase Structure): The leading coefficient of a row pivot (first nonzero number from the left) is called the pivot and is always strictly to the right of the pivot of the row above it. Therefore, any equation system in row-echelon form always has a “staircase” structure.

Definition: (Row-Echelon Form). A matrix is in row-echelon form if

- All rows that contain only zeros are at the bottom of the matrix; correspondingly, all rows that contain at least one nonzero element are on top of rows that contain only zeros.
- Looking at nonzero rows only, the first nonzero number from the left (also called the pivot or the leading coefficient) is always strictly to the right of the pivot of the row above it.

> (Basic and free variables): The variables corresponding to the pivots in the row-echelon form are called basic variables and the other variables are free variables.

> (Obtaining a particular solution): The row-echelon form makes our lives easier when we need to determine a particular solution. To do this, we express the right-hand side of the equation system using the pivot columns, such that
> $$b = \sum_{i=1}^{P}\lambda_i p_i$$
> where, pi, i = 1, 2, ..., P are the pivot columns. The lambda_i are determined easiest if we start from the right most pivot column and work our way to the left.

##### Step 1: Create the matrix

The system is:

$$
\begin{aligned}
-2x_1 + 4x_2 - 2x_3 - x_4 + 4x_5 &= -3, \\
4x_1 - 8x_2 + 3x_3 - 3x_4 + x_5 &= 2, \\
x_1 - 2x_2 + x_3 - x_4 + x_5 &= 0, \\
x_1 - 2x_2 - 3x_4 + 4x_5 &= a.
\end{aligned}
$$

This corresponds to the augmented matrix:

$$
\begin{bmatrix}
-2 & 4 & -2 & -1 & 4 & | & -3 \\
4 & -8 & 3 & -3 & 1 & | & 2 \\
1 & -2 & 1 & -1 & 1 & | & 0 \\
1 & -2 & 0 & -3 & 4 & | & a
\end{bmatrix}
$$

##### Step 2: Perform Gaussian elimination (Row-Echelon Form)

Now, we apply row operations to convert the matrix to row-echelon form.

1. First, we scale row 1 so that the leading entry becomes 1.

$$
\text{R}_1 \rightarrow \frac{1}{-2} \times \text{R}_1
$$

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
4 & -8 & 3 & -3 & 1 & | & 2 \\
1 & -2 & 1 & -1 & 1 & | & 0 \\
1 & -2 & 0 & -3 & 4 & | & a
\end{bmatrix}
$$

2. Eliminate the entries below the leading 1 in the first column by row operations.

$$
\text{R}_2 \rightarrow \text{R}_2 - 4 \times \text{R}_1, \quad \text{R}_3 \rightarrow \text{R}_3 - \text{R}_1, \quad \text{R}_4 \rightarrow \text{R}_4 - \text{R}_1
$$

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & -1 & -5 & 9 & | & -4 \\
0 & 0 & 0 & -1.5 & 3 & | & -1.5 \\
0 & 0 & -1 & -3.5 & 6 & | & a - 1.5
\end{bmatrix}
$$

3. Scale the second row.

$$
\text{R}_2 \rightarrow -\text{R}_2
$$

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & 1 & 5 & -9 & | & 4 \\
0 & 0 & 0 & -1.5 & 3 & | & -1.5 \\
0 & 0 & -1 & -3.5 & 6 & | & a - 1.5
\end{bmatrix}
$$

4. Eliminate the third row using the second row.

$$
\text{R}_4 \rightarrow \text{R}_4 + \text{R}_2
$$

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & 1 & 5 & -9 & | & 4 \\
0 & 0 & 0 & -1.5 & 3 & | & -1.5 \\
0 & 0 & 0 & 1.5 & -3 & | & a + 2.5
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & 1 & 5 & -9 & | & 4 \\
0 & 0 & 0 & -1.5 & 3 & | & -1.5 \\
0 & 0 & 0 & 1.5 & -3 & | & a + 2.5
\end{bmatrix}
$$

5. Eliminate the third row

We will work to eliminate the third row entry below the pivot in the fourth column.

Scale the third row:

$$
\text{R}_3 \rightarrow \frac{1}{-1.5} \times \text{R}_3 = \text{R}_3 \rightarrow \begin{bmatrix} 0 & 0 & 0 & 1 & -2 & | & 1 \end{bmatrix}
$$

This results in the following matrix:

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & 1 & 5 & -9 & | & 4 \\
0 & 0 & 0 & 1 & -2 & | & 1 \\
0 & 0 & 0 & 1.5 & -3 & | & a + 2.5
\end{bmatrix}
$$

Eliminate the entry in the fourth row (R₄) using R₃:

$$
\text{R}_4 \rightarrow \text{R}_4 - 1.5 \times \text{R}_3
$$

This gives us:

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & 1 & 5 & -9 & | & 4 \\
0 & 0 & 0 & 1 & -2 & | & 1 \\
0 & 0 & 0 & 0 & 0 & | & a + 1
\end{bmatrix}
$$

##### Step 3: Final Row-Echelon Form

Now the matrix is in row-echelon form (REF).

$$
\begin{bmatrix}
1 & -2 & 1 & 0.5 & -2 & | & 1.5 \\
0 & 0 & 1 & 5 & -9 & | & 4 \\
0 & 0 & 0 & 1 & -2 & | & 1 \\
0 & 0 & 0 & 0 & 0 & | & a + 1
\end{bmatrix}
$$

##### Step 4: Identify Pivot Columns

The pivot columns correspond to the leading entries in each row. In this case, the pivot columns are:

- Column 1 (for variable $x_1$)
- Column 3 (for variable $x_3$)
- Column 4 (for variable $x_4$)

If $a + 1 \neq 0$, the system is consistent. If $a + 1 = 0$, the system would have no solution (inconsistent).

Step 8: Solve the System

Now, let's back-substitute to find the values of the unknowns.

From row 3:

$$
x_4 - 2x_5 = 1 \implies x_4 = 1 + 2x_5
$$

From row 2:

$$
x_3 + 5x_4 - 9x_5 = 4
$$

Substitute $x_4 = 1 + 2x_5$:

$$
x_3 + 5(1 + 2x_5) - 9x_5 = 4
$$

$$
x_3 + 5 + 10x_5 - 9x_5 = 4 \implies x_3 = -1 - x_5
$$

From row 1:

$$
x_1 - 2x_2 + x_3 + 0.5x_4 - 2x_5 = 1.5
$$

Substitute $x_3 = -1 - x_5$ and $x_4 = 1 + 2x_5$:

$$
x_1 - 2x_2 + (-1 - x_5) + 0.5(1 + 2x_5) - 2x_5 = 1.5
$$

$$
x_1 - 2x_2 - 1 - x_5 + 0.5 + x_5 - 2x_5 = 1.5
$$

Simplifying:

$$
x_1 - 2x_2 - 0.5 - 2x_5 = 1.5 \implies x_1 = 2 + 2x_2 + 2x_5
$$

General Solution

Thus, the general solution is:

$$
x_1 = 2 + 2x_2 + 2x_5
$$

$$
x_2 = \text{free}
$$

$$
x_3 = -1 - x_5
$$

$$
x_4 = 1 + 2x_5
$$

$$
x_5 = \text{free}
$$

The system has two free variables, $x_2$ and $x_5$, indicating that there are infinitely many solutions depending on the values of these variables.

> (Reduced Row Echelon Form). An equation system is in reduced row-echelon form (also: row-reduced echelon form or row canonical form) if

- It is in row-echelon form.
- Every pivot is 1.
- The pivot is the only nonzero entry in its column.

#### Minus-1 Trick

$$
A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}
$$

We now augment this matrix to a $5 \times 5$ matrix by adding rows of the form (2.52) at the places where the pivots on the diagonal are missing and obtain:

$$
\tilde{A} = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & -1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4 \\
0 & 0 & 0 & 0 & -1
\end{bmatrix}
$$

From this form, we can immediately read out the solutions of $Ax = 0$ by taking the columns of $\tilde{A}$, which contain $-1$ on the diagonal:

$$
\left\{ x \in \mathbb{R}^5 : x = \lambda_1
\begin{bmatrix}
3 \\
-1 \\
0 \\
0 \\
0
\end{bmatrix}
+ \lambda_2
\begin{bmatrix}
3 \\
0 \\
9 \\
-4 \\
-1
\end{bmatrix}
, \, \lambda_1, \lambda_2 \in \mathbb{R} \right\}
$$

## Eigen Values and Eigen vectors

The solutions to the characteristic polynomial given by the equation

$$
| A - \lambda I | = 0
$$

where $\lambda$ are the eigen values such that, for a symetric matrix of order $n \times n$ has n eigen values such that

$$
Av = \lambda v \\
\text{    where v is a column vector and } \lambda \text{ is a eigen value.}
$$

### Repeating Eigen Values

If for a matrix A of dim $n \times n$ you get $n-2$ distinct and eigen values, then the eigen vectors corresponding to these $n-1$ eigen values are linearly independent. The remaining two eigen values have the same value, i.e., $\lambda = a$, then the maximum number of linearly independent eigen vectors are 2 since it's algebric multiplicity is 2.

If you get $\lambda = 0$ as an eigen value, and it's algebric multiplicity is 1, then rank of A is $R(A) = n-1$. This means the rank of null space is determined by the number of 0 valued eigen values.

If $\lambda = 0$ has algebric multiplicity > 1. Say the algebric multiplicity of 0 valued eigen value is a, then the number of linearly independent eigen vectors correspoding to 0 valued eigen value will give the span of Null space of matrix A. So the rank of the Null space can be atmost a, i.e., the maximum rank of null space can be the algebric multiplicity of the 0 valued eigen value.

> [REF](https://youtu.be/8LT1TK2mSCI?list=PLgjejdknTfWP9cYIHjxBRRdtGcUBLmCQC)

**Important Points**

1. For any symmetric matrix, the number of non-real eigen values are always divisible by 2. 0 (mod 2) = 0

## Matrix Decomposition

### LU Decomposition

LU decomposition is a matrix factorization method where a matrix $A$ is decomposed into the product of a **lower triangular matrix** $L$ and an **upper triangular matrix** $U$:

$$
A = LU
$$

Where:

- $L$: Lower triangular matrix with ones on the diagonal.
- $U$: Upper triangular matrix.

If **pivoting** is required to ensure numerical stability, the decomposition becomes:

$$
PA = LU
$$

Where $P$ is a permutation matrix.

#### Steps for LU Decomposition Without Pivoting

Let $A$ be an $n \times n$ matrix. The steps are as follows:

1. **Initialize $L$ and $U$:**

   - Start with $L = I$ (identity matrix) and $U = A$.

2. **Perform Gaussian Elimination:**
   - For each pivot $a_{kk}$, eliminate the elements below it in the $k$-th column by row operations.
   - Record the multipliers used for elimination in $L$.

#### Formula for Updating $L$ and $U$:

- Eliminate $a_{ik}$ ($i > k$) using:
  $$
  l_{ik} = \frac{a_{ik}}{a_{kk}} \quad \text{(store in $L$)}.
  $$
- Update $U$ for row $i$ as:
  $$
  u_{ij} = a_{ij} - l_{ik} \cdot u_{kj}, \quad j = k, k+1, \ldots, n.
  $$

3. **Repeat for All Rows:**
   - Continue this process for each column until $U$ becomes upper triangular.

#### Example

Consider the matrix:

$$
A = \begin{bmatrix}
2 & 3 & 1 \\
4 & 7 & 2 \\
6 & 18 & 5
\end{bmatrix}
$$

**Step 1: Start with $L = I$ and $U = A$:**<br>

$$
L = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix},
\quad
U = \begin{bmatrix}
2 & 3 & 1 \\
4 & 7 & 2 \\
6 & 18 & 5
\end{bmatrix}.
$$

**Step 2: Eliminate Below $a_{11}$ (First Column)**<br>

1. Compute multipliers:
   $$
   l_{21} = \frac{4}{2} = 2, \quad l_{31} = \frac{6}{2} = 3.
   $$
2. Update $U$ by subtracting multiples of the first row:

   $$
   \text{Row 2: } \begin{bmatrix} 4 & 7 & 2 \end{bmatrix} - 2 \cdot \begin{bmatrix} 2 & 3 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}.
   $$

   $$
   \text{Row 3: } \begin{bmatrix} 6 & 18 & 5 \end{bmatrix} - 3 \cdot \begin{bmatrix} 2 & 3 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 9 & 2 \end{bmatrix}.
   $$

   Updated $U$:

   $$
   U = \begin{bmatrix}
   2 & 3 & 1 \\
   0 & 1 & 0 \\
   0 & 9 & 2
   \end{bmatrix}.
   $$

3. Update $L$:
   $$
   L = \begin{bmatrix}
   1 & 0 & 0 \\
   2 & 1 & 0 \\
   3 & 0 & 1
   \end{bmatrix}.
   $$

**Step 3: Eliminate Below $a_{22}$ (Second Column)**<br>

1. Compute the multiplier:
   $$
   l_{32} = \frac{9}{1} = 9.
   $$
2. Update $U$:

   $$
   \text{Row 3: } \begin{bmatrix} 0 & 9 & 2 \end{bmatrix} - 9 \cdot \begin{bmatrix} 0 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 2 \end{bmatrix}.
   $$

   Updated $U$:

   $$
   U = \begin{bmatrix}
   2 & 3 & 1 \\
   0 & 1 & 0 \\
   0 & 0 & 2
   \end{bmatrix}.
   $$

3. Update $L$:
   $$
   L = \begin{bmatrix}
   1 & 0 & 0 \\
   2 & 1 & 0 \\
   3 & 9 & 1
   \end{bmatrix}.
   $$

$$
A = LU = \begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
3 & 9 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
2 & 3 & 1 \\
0 & 1 & 0 \\
0 & 0 & 2
\end{bmatrix}.
$$

#### Applications of LU Decomposition

1. **Solving Systems of Equations**: Solve $A\mathbf{x} = \mathbf{b}$ efficiently by solving $L\mathbf{y} = \mathbf{b}$ and $U\mathbf{x} = \mathbf{y}$.
2. **Matrix Inversion**: Invert $A$ by inverting $L$ and $U$.
3. **Determinants**: The determinant of $A$ is the product of the diagonal elements of $U$:
   $$
   \det(A) = \prod_{i} u_{ii}.
   $$

### QR Decomposition

### Eigen Decomposition

### Singular Value Decomposition

1. **Start with a matrix $A$:**

   $$
   A \in \mathbb{R}^{m \times n}
   $$

2. **Compute the eigenvalues and eigenvectors of $AA^T$:**

   - Let $U$ be the matrix of eigenvectors of $AA^T$.
   - Let $\Sigma$ be the diagonal matrix of the square roots of the eigenvalues of $AA^T$.

3. **Compute the eigenvalues and eigenvectors of $A^TA$:**

   - Let $V$ be the matrix of eigenvectors of $A^TA$.

4. **Construct the decomposition:**
   - $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix.
   - $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix with non-negative real numbers on the diagonal.
   - $V^T \in \mathbb{R}^{n \times n}$ is an orthogonal matrix.
   - Combine these to get: $A = U \Sigma V^T$.

### Applications of SVD:

- **Data compression:** Reducing the dimensionality of datasets.
- **Noise reduction:** Improving signal quality by removing noise.
- **Recommendation systems:** Helping to predict user preferences in systems like Netflix or Amazon.

**Note**<br>
For a symmetric matrix $A$, the singular values from the SVD are same as the mod of the eigen values of the matrix $A$.

## Special Matrices

### Symmetric Matrix

A **symmetric matrix** is a square matrix $A$ that satisfies the condition:

$$
A^T = A
$$

This means that the matrix is equal to its transpose. In other words, the elements satisfy:

$$
a_{ij} = a_{ji} \quad \forall i, j
$$

#### Properties

1. **Diagonal Elements**:  
   The diagonal elements of a symmetric matrix can be any real number.

2. **Eigenvalues**:

   - All eigenvalues of a symmetric matrix are real.
   - The matrix can be diagonalized using an orthogonal matrix.

3. **Orthogonality**:  
   The eigenvectors of a symmetric matrix corresponding to distinct eigenvalues are orthogonal.

   - Mathematically, if $A$ is a symmetric matrix, and $\lambda_1, \lambda_2$ are distinct eigenvalues with eigenvectors $v_1$ and $v_2$, then:
     $$
     v_1^T v_2 = 0
     $$

4. **Orthogonal Basis**:  
   The eigenvectors of a symmetric matrix form an orthogonal basis for the vector space $\mathbb{R}^n$. If normalized, they form an **orthonormal basis**.

5. **Real Eigenvectors**:  
   Just like the eigenvalues, the eigenvectors of a symmetric matrix are real.

6. **Diagonalization**:  
   A symmetric matrix $A$ can be diagonalized using its eigenvectors:

   $$
   A = Q \Lambda Q^T
   $$

   where:

   - $Q$ is an orthogonal matrix containing the eigenvectors of $A$ as columns ($Q^T Q = I$).
   - $\Lambda$ is a diagonal matrix containing the eigenvalues of $A$.

7. **Multiplicative Properties**:  
   If $A$ is symmetric and $v$ is an eigenvector, then for any integer $k$:

   $$
   A^k v = \lambda^k v
   $$

   where $\lambda$ is the eigenvalue corresponding to $v$.

8. **Spectral Theorem**:  
   For any symmetric matrix $A$, we can decompose it as:

   $$
   A = \sum\_{i=1}^n \lambda_i v_i v_i^T
   $$

   where:

   - $\lambda_i$ are the eigenvalues.
   - $v_i$ are the normalized eigenvectors.

9. **Orthogonal Diagonalization**:  
   A symmetric matrix can be written as:

   $$
   A = Q \Lambda Q^T
   $$

   where:

   - $Q$ is an orthogonal matrix ($Q^T Q = I$).
   - $\Lambda$ is a diagonal matrix with eigenvalues of $A$ on the diagonal.

10. **Positive Definiteness**:  
    A symmetric matrix $A$ is:

    - **Positive definite** if $x^T A x > 0$ for all $x \neq 0$.
    - **Positive semi-definite** if $x^T A x \geq 0$ for all $x$.
    - **Negative definite** if $x^T A x < 0$ for all $x \neq 0$.

11. **Symmetry and Determinant**:  
    The determinant of a symmetric matrix can be calculated using its eigenvalues:

    $$
    \det(A) = \prod_{i=1}^n \lambda_i
    $$

    where $\lambda_i$ are the eigenvalues.

12. **Rank**:  
    The rank of a symmetric matrix equals the number of nonzero eigenvalues.

13. **Additive and Multiplicative Symmetry**:

    - The sum of two symmetric matrices is symmetric:
      $$
      A + B \quad \text{is symmetric if } A^T = A \text{ and } B^T = B.
      $$
    - The product of two symmetric matrices is symmetric **if and only if they commute**:
      $$
      AB = BA \implies AB \text{ is symmetric.}
      $$

14. **Block Symmetry**:  
    If $A$ is a symmetric matrix, any block matrix partitioned as:
    $$
    A = \begin{bmatrix}
    B & C \\
    C^T & D
    \end{bmatrix}
    $$
    satisfies $C = C^T$ (i.e., the off-diagonal blocks must also be symmetric).

#### Example

Consider the symmetric matrix:

$$
A = \begin{bmatrix}
4 & 1 \\
1 & 3
\end{bmatrix}
$$

1. **Find Eigenvalues**: Solve $\det(A - \lambda I) = 0$:

   $$
   \begin{vmatrix}
   4 - \lambda & 1 \\
   1 & 3 - \lambda
   \end{vmatrix}
   = 0
   $$

   $$
   (4 - \lambda)(3 - \lambda) - 1 = 0 \implies \lambda^2 - 7\lambda + 11 = 0
   $$

2. **Eigenvalues**:

   $$
   \lambda_1 = 5, \quad \lambda_2 = 2
   $$

3. **Eigenvectors**:  
   For $\lambda_1 = 5$:

   $$
   (A - 5I)v = 0 \implies \begin{bmatrix}
   -1 & 1 \\
   1 & -2
   \end{bmatrix}
   v = 0
   $$

   Eigenvector: $v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

   For $\lambda_2 = 2$:

   $$
   (A - 2I)v = 0 \implies \begin{bmatrix}
   2 & 1 \\
   1 & 1
   \end{bmatrix}
   v = 0
   $$

   Eigenvector: $v_2 = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$

4. **Orthonormalize**: Normalize eigenvectors to form $Q$:

   $$
   Q = \begin{bmatrix}
   \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{5}} \\
   \frac{1}{\sqrt{2}} & \frac{2}{\sqrt{5}}
   \end{bmatrix}
   $$

5. **Diagonalization**:
   $$
   A = Q \Lambda Q^T
   $$

#### Applications

1. **Identify Symmetry Quickly**:  
   In a square matrix $A$, check $a_{ij} = a_{ji}$. This visual inspection often reveals symmetry.

2. **Use Eigenvalues for Determinants and Inverses**:  
   If eigenvalues are given:

   - Compute the determinant as the product of eigenvalues.
   - Compute the inverse using eigenvalues:
     $$
     A^{-1} = Q \Lambda^{-1} Q^T, \quad \text{where } \Lambda^{-1} = \text{diag}(\frac{1}{\lambda_1}, \dots, \frac{1}{\lambda_n}).
     $$

3. **Diagonalization**:  
   Use orthogonal diagonalization to simplify matrix operations like powers and exponentials:

   $$
   A^k = Q \Lambda^k Q^T, \quad \text{where } \Lambda^k \text{ is } \text{diag}(\lambda_1^k, \dots, \lambda_n^k).
   $$

4. **Focus on Quadratic Forms**:  
   Symmetric matrices often appear in quadratic forms $x^T A x$. To determine the definiteness of $A$:

   - Check the eigenvalues:  
     $\lambda > 0$ for positive definite, $\lambda \geq 0$ for semi-definite.

5. **Trace and Norms**:

   - The trace of a symmetric matrix equals the sum of its eigenvalues:
     $$
     \text{trace}(A) = \sum_{i=1}^n \lambda_i.
     $$
   - The Frobenius norm is:
     $$
     \|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}.
     $$

6. **Special Case: Identity Matrix**:  
   The identity matrix $I$ is symmetric, and its eigenvalues are all $1$. Use this property to simplify symmetric matrix equations.

7. **Projection and Symmetry**:  
   If $P$ is symmetric and $P^2 = P$, it’s a projection matrix. Use this property to simplify problems in linear regression or geometry.

#### Tips and Tricks for Competitive Exams

1. **Orthogonality Check**:  
   Use the orthogonality property to verify eigenvectors. If $v_1$ and $v_2$ are eigenvectors of a symmetric matrix, check:

   $$
   v_1^T v_2 = 0
   $$

2. **Simplify with Diagonalization**:  
   For powers of $A$, use the diagonalization formula:

   $$
   A^k = Q \Lambda^k Q^T
   $$

   This reduces the computational effort significantly.

3. **Test Real Eigenvalues**:  
   In multiple-choice questions, if the matrix is symmetric, reject any option with complex eigenvalues.

4. **Spectral Decomposition**:  
   For quadratic forms $x^T A x$, decompose $A$ using eigenvalues and eigenvectors:

   $$
   x^T A x = \sum\_{i=1}^n \lambda_i (x^T v_i)^2
   $$

5. **Eigenvector Matrix $Q$**:  
   If eigenvectors are normalized and form an orthogonal matrix $Q$, verify:

   $$
   Q^T Q = I
   $$

6. **Quick Eigenvalue-Eigenvector Pairing**:  
   When $A v = \lambda v$:
   - Test eigenvectors by substitution into the equation $A v - \lambda v = 0$.

<hr>

### Idempotent Matrix

An **idempotent matrix** is a square matrix $A$ that satisfies the equation:

$$
A^2 = A
$$

This means that when the matrix is multiplied by itself, the result is the same as the original matrix.

#### Properties

1. **Eigenvalues**:  
   The eigenvalues of an idempotent matrix are either $0$ or $1$.

   - If $A$ is an $n \times n$ idempotent matrix, the eigenvalue equation is:
     $$
     Av = \lambda v, \quad \text{and } A^2v = Av \implies \lambda^2 = \lambda \implies \lambda = 0 \text{ or } \lambda = 1
     $$

2. **Trace**:  
   The trace of an idempotent matrix equals the rank of the matrix:

   $$
   \text{trace}(A) = \text{rank}(A)
   $$

3. **Determinant**:  
   The determinant of an idempotent matrix is either $0$ or $1$:

   $$
   \det(A) = 0 \quad \text{or} \quad \det(A) = 1
   $$

4. **Symmetry**:

   - If $A$ is symmetric and idempotent, it represents an orthogonal projection matrix.

5. **Rank**:

   - The rank of $A$ equals the trace of $A$. If $A$ is $n \times n$, then:
     $$
     0 \leq \text{rank}(A) \leq n
     $$

6. **Nilpotency**:
   - If $A$ is idempotent and $A \neq I$, $A$ is not invertible unless $A = I$.

> Idempotent matrices $A$ and $B$, have the following properties:
>
> 1. $A^n = A$, for n = 1, 2, 3, ...
> 2. $I - A$ is idempotent
> 3. $A^H$ is idempotent
> 4. $I - A^H$ is idempotent
> 5. If $AB = BA$ ⇒ $AB$ is idempotent
> 6. $rank(A) = Tr(A)$
> 7. $A(I - A) = 0$
> 8. $(I - A)A = 0$
> 9. $A^+ = A$
> 10. $f(sI + tA) = (I - A)f(s) + Af(s + t)$ <br>
>     Note that $A - I$ is not necessarily idempotent.

#### Examples

1. A trivial idempotent matrix:

   $$
   A = \begin{bmatrix}
   1 & 0 \\
   0 & 1
   \end{bmatrix}, \quad A^2 = A
   $$

2. A non-identity idempotent matrix:
   $$
   A = \begin{bmatrix}
   1 & 0 \\
   0 & 0
   \end{bmatrix}, \quad A^2 = A
   $$

#### Applications

1. **Squaring the matrix**:  
   Check if $A^2 = A$.

2. **Eigenvalues**:  
   If the eigenvalues are only $0$ and $1$, $A$ might be idempotent.

3. **Rank and trace**:  
   Verify if the rank equals the trace.

4. **Projection matrices**:  
   In some contexts, idempotent matrices arise naturally as projection operators. If $A^2 = A$ and $A$ is symmetric, it represents a projection.

**Tricks to Identify an Idempotent Matrix**

1. **Spectral decomposition**:  
   For a symmetric idempotent matrix, $A$, we can write:

   $$
   A = Q \Lambda Q^T
   $$

   where $\Lambda$ contains $0$ and $1$ on its diagonal, and $Q$ is an orthogonal matrix.

2. **Addition and subtraction**:

   - If $A$ and $B$ are idempotent and commute ($AB = BA$), then:
     $$
     (A + B)^2 = A + B \quad \text{and} \quad (A - B)^2 = A - B
     $$

3. **Diagonalization**:  
   Any symmetric idempotent matrix can be diagonalized. Its diagonal entries are the eigenvalues $0$ or $1$.

4. **Projection in Linear Models**:  
   In regression analysis, the hat matrix $H$ is idempotent:

   $$
   H = X(X^TX)^{-1}X^T, \quad H^2 = H
   $$

5. **Data Transformation**:  
   Idempotent matrices are used to project data points onto subspaces.

## Misc

**Similarity of Matrices**<br>
Similar matrices are square matrices that represent the same linear map, but under different bases. They have the same rank, trace, determinant, and eigenvalues.
Explanation

- Two matrices $A$ and $B$ are similar if there exists a nonsingular matrix $S$ such that $A=S^{-1}BS$. 
- The transformation $A\mapsto P^{-1}AP$ is called a similarity transformation or conjugation of the matrix $A$. 
- Similar matrices have similar properties, even though they are not equal. 
- Similarity is an equivalence relation, which means it satisfies the properties of transitivity. 

Two matrices $A$ and $B$ are called similar if there exists another matrix $S$ such that

$$
 S^{-1}AS = B.
$$

Consider the statements:

I. If $A$ and $B$ are similar then they have identical rank.

II. If $A$ and $B$ are similar then they have identical trace.

III. $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ and $B = \begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix}$ are similar.

## References

1. [Linear Algebra by 3Blue1Brown](https://www.3blue1brown.com/topics/linear-algebra)
2. [MIT - Strang Video Lectures](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
3. [MIT-Gilbert Strang LA Book and other material](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html)

## Books

1. [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
2. Deisenroth, M. P., Faisal, A. A., &#38; Ong, C. S. (2020). Mathematics for Machine Learning. Cambridge: Cambridge University Press.
3. Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
4. Strang, G. (2021). Introduction to Linear Algebra (5th ed.). Cambridge: CUP.

---

## Appendix

<span align='center'>

### Notation and Nomenclature

| Symbol            | Meaning                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| $A\quad$          | Matrix                                                                               |
| $A_{ij}\quad$     | Matrix indexed for some purpose                                                      |
| $A_{i}\quad$      | Matrix indexed for some purpose                                                      |
| $A_{ij}^{~}\quad$ | Matrix indexed for some purpose                                                      |
| $A^{n}\quad$      | The nth power of a square matrix                                                     |
| $A^{-1}\quad$     | The inverse matrix of the matrix $A$                                                 |
| $A^{+}\quad$      | The pseudo inverse matrix of the matrix $A\quad$                                     |
| $A^{1/2}\quad$    | The square root of a matrix (if unique), not elementwise                             |
| $(A)_{ij}\quad$   | The (i,j)th entry of the matrix $A$                                                  |
| $[A]_{ij}\quad$   | The ij-submatrix, i.e., $A\quad$ : with ith row and jth column deleted               |
| $a\quad$          | Vector (column-vector)                                                               |
| $a_{i}\quad$      | Vector indexed for some purpose                                                      |
| $a_{i}\quad$      | The ith element of the vector $a$                                                    |
| $a\quad$          | Scalar                                                                               |
| $ℜz\quad$         | Real part of a scalar                                                                |
| $ℜz\quad$         | Real part of a vector                                                                |
| $ℜZ\quad$         | Real part of a matrix                                                                |
| $ℑz\quad$         | Imaginary part of a scalar                                                           |
| $ℑz\quad$         | Imaginary part of a vector                                                           |
| $ℑZ\quad$         | Imaginary part of a matrix                                                           |
| $det(A)\quad$     | Determinant of $A$                                                                   |
| $Tr(A)\quad$      | Trace of the matrix $A$                                                              |
| $diag(A)\quad$    | Diagonal matrix of the matrix $A$, i.e., $(diag(A))_{jj} = δ_{ij}A_{ij}$             |
| $eig(A)\quad$     | Eigenvalues of the matrix $A$                                                        |
| $vec(A)\quad$     | The vector-version of the matrix $A\quad$                                            |
| $sup\quad$        | Supremum of a set                                                                    |
| $\|\|A\|\|\quad$  | Matrix norm (subscript if any denotes what norm)                                     |
| $A^{T}\quad$      | Transposed matrix                                                                    |
| $A^{-T}\quad$     | The inverse of the transposed and vice versa, $A^{-T} = (A^{-1})^{T} = (A^{T})^{-1}$ |
| $A^{*}\quad$      | Complex conjugated matrix                                                            |
| $A^{H}\quad$      | Transposed and complex conjugated matrix (Hermitian)                                 |
| $A ∘ B\quad$      | Hadamard (elementwise) product                                                       |
| $A ⊗ B\quad$      | Kronecker product                                                                    |
| $O\quad$          | The null matrix. Zero in all entries.                                                |
| $I\quad$          | The identity matrix                                                                  |
| $J^{ij}\quad$     | The single-entry matrix, 1 at (i,j) and zero elsewhere                               |
| $Σ\quad$          | A positive definite matrix                                                           |
| $Λ\quad$          | A diagonal matrix                                                                    |

</span>
</span>
