In this assignment, we aim to solve a regularized regression problem that promotes sparsity—not at the level of individual predictors, but at the level of predefined groups of predictors. This is particularly useful when predictors are naturally grouped and we want to select or discard entire groups based on their relevance to the prediction task.

Objective
We are given a dataset related to Parkinson's disease and asked to predict the total symptom score (UPDRS) using voice measurements and demographic features. The goals are to:

Implement ridge regression using stochastic gradient descent.

Implement group LASSO using proximal gradient descent.

Compare the performance and sparsity patterns of these methods.

Explore acceleration techniques for faster convergence.

Dataset Description
The dataset contains:

𝑁
=
5785
N=5785 observations.

𝑝
=
18
p=18 predictors (features), stored in X_train.csv.

The target variable is the total UPDRS score, stored in y_train.csv.

The predictors are grouped as follows:

Demographics: age, sex

Jitter Features: Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP

Shimmer Features: Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA

Noise-to-Harmonics Ratio: NHR, HNR

Nonlinear and Dynamical Features: RPDE, DFA, PPE

(a) Ridge Regression
We solve the ridge regression problem:

min
⁡
𝛽
∈
𝑅
𝑝
+
1
1
2
𝑁
∥
𝑋
𝛽
+
𝑏
−
𝑦
∥
2
+
𝜆
∥
𝛽
∥
2
β∈R 
p+1
 
min
​
  
2N
1
​
 ∥Xβ+b−y∥ 
2
 +λ∥β∥ 
2
 
where 
𝑏
b is the bias term, treated separately from the regularized parameters 
𝛽
β.

i. Gradient Updates
The gradients of the objective with respect to 
𝛽
β and 
𝑏
b are:

∇
𝛽
𝑓
(
𝛽
,
𝑏
)
=
𝑋
⊤
(
𝑋
𝛽
+
𝑏
−
𝑦
)
+
2
𝜆
𝛽
∇ 
β
​
 f(β,b)=X 
⊤
 (Xβ+b−y)+2λβ
∇
𝑏
𝑓
(
𝛽
,
𝑏
)
=
1
⊤
(
𝑋
𝛽
+
𝑏
−
𝑦
)
∇ 
b
​
 f(β,b)=1 
⊤
 (Xβ+b−y)
The update rules are:

𝛽
(
𝑘
+
1
)
=
𝛽
(
𝑘
)
−
𝑡
⋅
∇
𝛽
𝑓
(
𝛽
(
𝑘
)
,
𝑏
(
𝑘
)
)
β 
(k+1)
 =β 
(k)
 −t⋅∇ 
β
​
 f(β 
(k)
 ,b 
(k)
 )
𝑏
(
𝑘
+
1
)
=
𝑏
(
𝑘
)
−
𝑡
⋅
∇
𝑏
𝑓
(
𝛽
(
𝑘
)
,
𝑏
(
𝑘
)
)
b 
(k+1)
 =b 
(k)
 −t⋅∇ 
b
​
 f(β 
(k)
 ,b 
(k)
 )
ii–iii. Implementation and Observations
The implementation was completed in the submitted code files. Based on the convergence plots for 16 different configurations, we observed:

The step size should be moderate: too small slows convergence, too large causes oscillation.

Smaller batch sizes generally improve convergence speed.

Larger batch sizes help stabilize training when step size is large, but slow convergence.

(b) Group LASSO
We solve the group LASSO problem:

min
⁡
𝛽
∈
𝑅
𝑝
+
1
1
2
𝑁
∥
𝑋
𝛽
+
𝑏
−
𝑦
∥
2
+
𝜆
∑
𝑗
=
1
𝐽
𝑤
𝑗
∥
𝛽
(
𝑗
)
∥
2
β∈R 
p+1
 
min
​
  
2N
1
​
 ∥Xβ+b−y∥ 
2
 +λ 
j=1
∑
J
​
 w 
j
​
 ∥β 
(j)
 ∥ 
2
​
 
where 
𝑤
𝑗
w 
j
​
  is the weight for group 
𝑗
j, typically set to the number of features in that group.

i. Proximal Operator
Using the group-wise proximal operator:

prox
𝜆
𝑤
𝑗
∥
⋅
∥
2
(
𝑣
(
𝑗
)
)
=
(
1
−
𝜆
𝑤
𝑗
∥
𝑣
(
𝑗
)
∥
2
)
+
𝑣
(
𝑗
)
prox 
λw 
j
​
 ∥⋅∥ 
2
​
 
​
 (v 
(j)
 )=(1− 
∥v 
(j)
 ∥ 
2
​
 
λw 
j
​
 
​
 ) 
+
​
 v 
(j)
 
The full proximal update is applied group-wise:

𝛽
(
𝑘
+
1
)
=
prox
𝑡
ℎ
(
𝛽
(
𝑘
)
−
𝑡
⋅
∇
𝑔
(
𝛽
(
𝑘
)
)
)
β 
(k+1)
 =prox 
th
​
 (β 
(k)
 −t⋅∇g(β 
(k)
 ))
ii–iii. Implementation and Group Selection
The implementation was completed in the submitted code files. Based on the output, the selected groups were:

RPDE

Age

PPE

iv. Comparison with LASSO
Group LASSO showed better sparsity and faster convergence compared to standard LASSO. The selected groups were more interpretable and consistent with domain knowledge.

v. Accelerated Proximal Gradient
We implemented Nesterov’s acceleration. The update rule for 
𝛽
β becomes:

𝛽
(
𝑘
+
1
)
=
prox
𝑡
ℎ
(
𝑧
(
𝑘
)
−
𝑡
⋅
∇
𝑔
(
𝑧
(
𝑘
)
)
)
β 
(k+1)
 =prox 
th
​
 (z 
(k)
 −t⋅∇g(z 
(k)
 ))
where 
𝑧
(
𝑘
)
z 
(k)
  is a momentum term. The bias term 
𝑏
b is updated separately.

vi. Observations
The accelerated algorithm converged significantly faster — approximately 10× closer to the optimal value compared to the unaccelerated method. It also preserved the sparsity structure effectively.
