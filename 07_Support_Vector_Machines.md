# Lecture 7: Support Vector Machines (SVM)

> **Course:** BITS F464 — Machine Learning
> **Covers:** SVM intuition (max-margin), support vectors, the margin formula, the optimal-hyperplane optimization, a **full worked Linear SVM example** (finding α, w, b step by step), Non-linear SVM (lifting, kernels), and a **full worked Kernel SVM example**.
>
> **How to read this:** Both worked examples are traced completely — every dot product, every α equation solved. These two examples are **the** SVM exam problems; memorize the 6-step procedure.

---

## Table of Contents

1. [SVM Introduction — the Max-Margin Idea](#1-svm-introduction--the-max-margin-idea)
2. [Margin & Support Vectors](#2-margin--support-vectors)
3. [The Margin Formula](#3-the-margin-formula)
4. [The Optimal Hyperplane (optimization problem)](#4-the-optimal-hyperplane-optimization-problem)
5. [⭐ Worked Example 1 — Linear SVM](#5--worked-example-1--linear-svm-full-trace)
6. [Non-Linear SVM — Lifting & Kernels](#6-non-linear-svm--lifting--kernels)
7. [⭐ Worked Example 2 — Kernel SVM](#7--worked-example-2--kernel-svm-full-trace)
8. [Advantages & Disadvantages](#8-advantages--disadvantages)
9. [Quick Revision Sheet](#9-quick-revision-sheet)

---

## 1. SVM Introduction — the Max-Margin Idea

A **linear discriminant function**: $g(x) = w^t x + w_0$
- $g(x) > 0 \Rightarrow$ class 1
- $g(x) < 0 \Rightarrow$ class 2

(Same as Lecture 3's discriminant — see [03_Discriminant_Functions.md](03_Discriminant_Functions.md).)

**The problem:** *many* lines can separate the two classes. Which is best?

> 🔑 **The SVM idea: choose the hyperplane that is AS FAR AS POSSIBLE from the closest examples** (maximize distance to the closest example).

**Why?** If the boundary is jammed right next to a training point, a *new* point near it will likely fall on the **wrong side** → misclassified. A boundary far from all points → new points near old ones get classified **correctly** → better generalization.

```
  BAD (line close to points)        GOOD (line far from points)
   ●●● │ ■■■                          ●●●   │   ■■■
   ●●●│■■■   ← new point near          ●●●   │   ■■■
   ●●●│ x■■    line → wrong side       ●●●   │   ■■■
       small distance                   ← larger distance →
```

**For the optimal hyperplane:** distance to the closest **negative** example = distance to the closest **positive** example (it sits exactly in the middle).

---

## 2. Margin & Support Vectors

> - **Margin** = twice the absolute distance $b$ of the closest example to the separating hyperplane (the "width" of the no-man's-land).
> - **Support vectors** = the training samples **closest to the hyperplane** (they "support"/define it).

```
        ╱ ← margin boundary (dashed)
   ●  ╱   b
   ● ╱─────── ← separating hyperplane (solid)
    ╱   b
   ╱ ■  ■        ● = class −,  ■ = class +
  ╱╲ margin = 2b   circled points touching dashed lines = SUPPORT VECTORS
```

> 🔑 **Key insight (exam):** Only the **support vectors** matter. If you removed all other points, the SVM boundary would be *identical*. This is why SVM is memory-efficient — it only stores the support vectors.

> 💡 **SVM = Maximum Margin Classifier.** Widest possible "street" between the two classes; support vectors are the points on the kerb.

---

## 3. The Margin Formula

**Absolute distance** of point $x$ from boundary $g(x)=0$:
$$\frac{|w^t x + w_0|}{\|w\|}$$

This distance is **scale-invariant**: scaling $g_1(x) = \alpha g(x)$ doesn't change it:
$$\frac{|\alpha w^t x + \alpha w_0|}{\|\alpha w\|} = \frac{|w^t x + w_0|}{\|w\|}$$

**For uniqueness**, we *normalize*: set $|w^t x_i + w_0| = 1$ for the closest example $x_i$ (a "support vector"). Then the distance from the closest sample to the boundary is:
$$\frac{|w^t x_i + w_0|}{\|w\|} = \frac{1}{\|w\|}$$

So the **margin** (both sides) is:

$$\boxed{\,m = \dfrac{2}{\|w\|}\,}$$

> 💡 **Crucial consequence:** maximizing the margin $\dfrac{2}{\|w\|}$ = **minimizing $\|w\|$**. Big $\|w\|$ → thin margin; small $\|w\|$ → fat margin.

---

## 4. The Optimal Hyperplane (optimization problem)

**Maximize** $m = \dfrac{2}{\|w\|}$ **subject to** every point being on its correct side **outside** the margin:

$$w^t x_i + w_0 \geq +1 \quad \text{if } x_i \text{ is positive}$$
$$w^t x_i + w_0 \leq -1 \quad \text{if } x_i \text{ is negative}$$

Define labels $z_i = +1$ (positive) or $z_i = -1$ (negative). Both constraints combine into one neat form: $z_i(w^t x_i + w_0) \geq 1$.

Maximizing $\frac{2}{\|w\|}$ = minimizing $\frac12\|w\|^2$. So the SVM training problem is:

$$\boxed{\;\text{minimize } J(w) = \tfrac{1}{2}\|w\|^2 \quad \text{constrained to } z_i(w^t x_i + w_0) \geq 1 \;\;\forall i\;}$$

> 🔑 This is a **constrained quadratic optimization** problem (solved with Lagrange multipliers → the α's in the worked examples). The $\frac12$ and the square are just to make the calculus clean; minimizing $\|w\|^2$ ⟺ minimizing $\|w\|$ ⟺ maximizing the margin.

---

## 5. ⭐ Worked Example 1 — Linear SVM (full trace)

**This is the #1 exam problem.** Follow the 6 steps exactly.

**Given:**
- Positive points (+): (3,1), (3,−1), (6,1), (6,−1)
- Negative points (−): (1,0), (0,1), (0,−1), (−1,0)
- *Find a simple SVM that separates them.*

### Step 1 — Plot the data

```
  x₂
 1 ■        ●        ●          ● = "+" class
 0 ■   ■              (gap)     ■ = "−" class
-1 ■        ●        ●
   -1 0 1   3        6   → x₁
```
The "+" points are on the right, "−" on the left. Separable by a vertical-ish line.

### Step 2 — Find the support vectors (by observation)

The points **closest to the opposing class**:
$$s_1 = \begin{pmatrix}1\\0\end{pmatrix} \text{ (negative)}, \quad s_2 = \begin{pmatrix}3\\1\end{pmatrix}, \quad s_3 = \begin{pmatrix}3\\-1\end{pmatrix} \text{ (positive)}$$

### Step 3 — Augment each support vector with a bias input of 1

Add a 3rd component = 1 (this absorbs the bias $w_0$):
$$\tilde{s_1} = \begin{pmatrix}1\\0\\1\end{pmatrix}, \quad \tilde{s_2} = \begin{pmatrix}3\\1\\1\end{pmatrix}, \quad \tilde{s_3} = \begin{pmatrix}3\\-1\\1\end{pmatrix}$$

### Step 4–5 — Set up & solve the α equations

For each support vector, write $\sum_j \alpha_j (\tilde{s_j}\cdot \tilde{s_i}) = $ its label (−1 for negative, +1 for positive):

$$\alpha_1(\tilde{s_1}\!\cdot\!\tilde{s_1}) + \alpha_2(\tilde{s_2}\!\cdot\!\tilde{s_1}) + \alpha_3(\tilde{s_3}\!\cdot\!\tilde{s_1}) = -1$$
$$\alpha_1(\tilde{s_1}\!\cdot\!\tilde{s_2}) + \alpha_2(\tilde{s_2}\!\cdot\!\tilde{s_2}) + \alpha_3(\tilde{s_3}\!\cdot\!\tilde{s_2}) = +1$$
$$\alpha_1(\tilde{s_1}\!\cdot\!\tilde{s_3}) + \alpha_2(\tilde{s_2}\!\cdot\!\tilde{s_3}) + \alpha_3(\tilde{s_3}\!\cdot\!\tilde{s_3}) = +1$$

**Compute the dot products** (e.g. $\tilde{s_1}\cdot\tilde{s_1} = 1·1+0·0+1·1 = 2$):

| Dot product | Value |
|---|---|
| $\tilde{s_1}\cdot\tilde{s_1} = (1,0,1)\cdot(1,0,1)$ | $1+0+1 = 2$ |
| $\tilde{s_2}\cdot\tilde{s_1} = (3,1,1)\cdot(1,0,1)$ | $3+0+1 = 4$ |
| $\tilde{s_3}\cdot\tilde{s_1} = (3,-1,1)\cdot(1,0,1)$ | $3+0+1 = 4$ |
| $\tilde{s_2}\cdot\tilde{s_2} = (3,1,1)\cdot(3,1,1)$ | $9+1+1 = 11$ |
| $\tilde{s_3}\cdot\tilde{s_2} = (3,-1,1)\cdot(3,1,1)$ | $9-1+1 = 9$ |
| $\tilde{s_3}\cdot\tilde{s_3} = (3,-1,1)\cdot(3,-1,1)$ | $9+1+1 = 11$ |

Substituting gives the system:
$$2\alpha_1 + 4\alpha_2 + 4\alpha_3 = -1$$
$$4\alpha_1 + 11\alpha_2 + 9\alpha_3 = +1$$
$$4\alpha_1 + 9\alpha_2 + 11\alpha_3 = +1$$

**Solve:**
- Eqs 2 & 3 are symmetric → $\alpha_2 = \alpha_3$.
- Sub into Eq 2: $4\alpha_1 + 20\alpha_2 = 1$. From Eq 1: $2\alpha_1 + 8\alpha_2 = -1$.
- Solving these two: $\boxed{\alpha_1 = -3.5,\quad \alpha_2 = 0.75,\quad \alpha_3 = 0.75}$

### Step 6 — Recover the hyperplane

$$\tilde{w} = \sum_i \alpha_i \tilde{s_i} = -3.5\begin{pmatrix}1\\0\\1\end{pmatrix} + 0.75\begin{pmatrix}3\\1\\1\end{pmatrix} + 0.75\begin{pmatrix}3\\-1\\1\end{pmatrix}$$

Compute component-wise:
- 1st: $-3.5(1) + 0.75(3) + 0.75(3) = -3.5 + 2.25 + 2.25 = 1$
- 2nd: $-3.5(0) + 0.75(1) + 0.75(-1) = 0$
- 3rd: $-3.5(1) + 0.75(1) + 0.75(1) = -3.5 + 1.5 = -2$

$$\tilde{w} = \begin{pmatrix}1\\0\\-2\end{pmatrix}$$

The **last entry is the bias** $b$. So:
$$\boxed{\,w = \begin{pmatrix}1\\0\end{pmatrix}, \quad b = -2\,}$$

**Hyperplane:** $y = wx + b = 1\cdot x_1 + 0\cdot x_2 - 2 = 0$ → **$x_1 = 2$** (a vertical line at x=2).

✅ **Check:** Negative point (1,0): $1(1)+0(0)-2 = -1 < 0$ → class −. ✓ Positive point (3,1): $1(3)+0(1)-2 = +1 > 0$ → class +. ✓ The line $x_1=2$ perfectly splits left (−) from right (+).

> ✅ **Exam recipe (memorize):** ① plot → ② pick support vectors by eye → ③ augment with a "1" → ④ build $\sum\alpha_j(\tilde s_j\!\cdot\!\tilde s_i)=\pm1$ system → ⑤ solve for α → ⑥ $\tilde w=\sum\alpha_i\tilde s_i$; split into $w$ + bias $b$ (last entry).

---

## 6. Non-Linear SVM — Lifting & Kernels

**Problem:** Some data is **not linearly separable** in its original space.

**Example (slide 18):** points on a 1-D line, "−" in the middle, "+" on both ends — **no single point** can split them. But **map** them up using $\Phi(x) = (x, x^2)$ → in 2-D they form a **parabola** that a line *can* split!

```
  1-D (not separable):  ● ● ■ ■ ■ ■ ■ ● ●   ← can't split with one cut
                                ↓ Φ(x)=(x,x²)
  2-D (separable):       ●           ●        ← now a straight line
                          ●    ___  ●            separates them!
                           ●__/■■■\__●
```

> 🔑 **The lifting trick:** project data into a **higher-dimensional space** where it becomes linearly separable, then apply linear SVM there.

**The catch:** higher dimensions → **curse of dimensionality** (poor generalization + expensive computation).

**SVM avoids the curse by:**
1. **Large-margin enforcement** → good generalization even in high dimensions.
2. **Kernel functions** → the high-dimensional computation is done **implicitly** (never actually compute the high-dim coordinates).

### Why kernels?
- Make a non-separable problem **separable**.
- Map data into a better representational space — **without** paying the high-dim cost (the "kernel trick").

### Common kernels (memorize the formulas)

| Kernel | Formula |
|---|---|
| **Linear** | $k(x_i,x_j) = x_i\cdot x_j$ |
| **Polynomial** | $k(x_i,x_j) = (x_i\cdot x_j + 1)^d$ |
| **Gaussian** | $k(x,y) = \exp\left(-\dfrac{\|x-y\|^2}{2\sigma^2}\right)$ |
| **RBF** | $k(x_i,x_j) = \exp\left(-\gamma\|x_i-x_j\|^2\right)$ |

> 💡 **Kernel trick intuition:** a kernel computes the *dot product in the high-dim space* using only the *original* vectors — so you get the power of high dimensions without ever building them. Gaussian/RBF are the most common defaults.

#### 🧠 Worked example — polynomial kernel

$x_i=(1,2)$, $x_j=(3,1)$, degree $d=2$:
$$x_i\cdot x_j = 1(3)+2(1) = 5, \quad k = (5+1)^2 = 6^2 = \mathbf{36}$$

---

## 7. ⭐ Worked Example 2 — Kernel SVM (full trace)

Same 6-step procedure, but with a **mapping function applied first**.

**Given:**
- Positive (+): (2,2), (2,−2), (−2,−2), (−2,2)  *(the 4 outer corners)*
- Negative (−): (1,1), (1,−1), (−1,−1), (−1,1)  *(the 4 inner corners)*

```
  x₂
 2 ●       ●        ● = "+" (outer),  ■ = "−" (inner)
 1   ■   ■          NOT linearly separable — "−" is
 0      ┼            surrounded by "+" on all sides!
-1   ■   ■
-2 ●       ●
   -2 -1 0 1 2 → x₁
```

### Step 2 — Apply the mapping function

$$\Phi_1\begin{pmatrix}x_1\\x_2\end{pmatrix} = \begin{cases} \begin{pmatrix}4-x_2+|x_1-x_2|\\ 4-x_1+|x_1-x_2|\end{pmatrix} & \text{if } \sqrt{x_1^2+x_2^2} > 2 \\[3mm] \begin{pmatrix}x_1\\x_2\end{pmatrix} & \text{otherwise} \end{cases}$$

Apply it (outer points have radius > 2 so they transform; inner points stay):
- **Positive** (2,2)→(2,2), (2,−2)→(10,6), (−2,−2)→(6,6), (−2,2)→(6,10)
- **Negative** stay: (1,1), (1,−1), (−1,−1), (−1,1)

Now in the new space the classes **are linearly separable** (slide 24 plot confirms).

### Step 3 — Support vectors (by observation)

$$s_1 = \begin{pmatrix}1\\1\end{pmatrix}\text{ (negative)}, \quad s_2 = \begin{pmatrix}2\\2\end{pmatrix}\text{ (positive)}$$

### Step 4 — Augment with bias 1

$$\tilde{s_1} = \begin{pmatrix}1\\1\\1\end{pmatrix}, \quad \tilde{s_2} = \begin{pmatrix}2\\2\\1\end{pmatrix}$$

### Step 5 — Solve the α equations

$$\alpha_1(\tilde{s_1}\!\cdot\!\tilde{s_1}) + \alpha_2(\tilde{s_2}\!\cdot\!\tilde{s_1}) = -1$$
$$\alpha_1(\tilde{s_1}\!\cdot\!\tilde{s_2}) + \alpha_2(\tilde{s_2}\!\cdot\!\tilde{s_2}) = +1$$

Dot products:
- $\tilde{s_1}\cdot\tilde{s_1} = 1+1+1 = 3$
- $\tilde{s_2}\cdot\tilde{s_1} = 2+2+1 = 5$
- $\tilde{s_2}\cdot\tilde{s_2} = 4+4+1 = 9$

System:
$$3\alpha_1 + 5\alpha_2 = -1$$
$$5\alpha_1 + 9\alpha_2 = +1$$

**Solve:** from Eq1 $\alpha_1 = \frac{-1-5\alpha_2}{3}$. Substitute into Eq2 → $\boxed{\alpha_1 = -7,\quad \alpha_2 = 4}$

### Step 6 — Recover the hyperplane

$$\tilde{w} = \alpha_1\tilde{s_1} + \alpha_2\tilde{s_2} = -7\begin{pmatrix}1\\1\\1\end{pmatrix} + 4\begin{pmatrix}2\\2\\1\end{pmatrix}$$
- 1st: $-7 + 8 = 1$
- 2nd: $-7 + 8 = 1$
- 3rd: $-7 + 4 = -3$

$$\tilde{w} = \begin{pmatrix}1\\1\\-3\end{pmatrix} \;\Rightarrow\; \boxed{w = \begin{pmatrix}1\\1\end{pmatrix}, \quad b = -3}$$

**Hyperplane:** $x_1 + x_2 - 3 = 0$ in the *mapped* space — which corresponds to a **non-linear** boundary (a circle/curve) back in the original space, perfectly separating inner from outer points.

> 🔑 **The big idea:** the *same linear SVM procedure*, just preceded by a mapping/kernel step. The boundary is linear in the lifted space but **non-linear** in the original space — that's how SVM handles non-separable data.

---

## 8. Advantages & Disadvantages

| ✅ Advantages | ❌ Disadvantages |
|---|---|
| Accurate in **high-dimensional** space | Prone to **overfitting** (wrong kernel) |
| **Different kernels** for different problems | **No probability estimation** (just class) |
| **Memory efficient** — only stores support vectors | Efficient only for **small/average** datasets |
| Strong generalization (max margin) | **Bad if features > samples** |

> 🔑 **Exam one-liners:** SVM = max-margin classifier defined only by support vectors. Strength = high-dim accuracy + kernels. Weakness = slow on big data, no probabilities, sensitive to kernel choice.

---

## 9. Quick Revision Sheet

**SVM idea:** pick the separating hyperplane with the **maximum margin** (farthest from closest points) → best generalization.

**Support vectors** = closest points to the boundary; they alone define it (others irrelevant → memory efficient).

**Margin** $m = \dfrac{2}{\|w\|}$. Maximize margin ⟺ **minimize $\|w\|$** ⟺ minimize $\frac12\|w\|^2$.

**Optimization:** minimize $\frac12\|w\|^2$ s.t. $z_i(w^tx_i+w_0)\geq1$ (z=+1 pos, −1 neg).

**6-step worked procedure:** ① plot ② pick SVs ③ augment with 1 ④ build $\sum\alpha_j(\tilde s_j\!\cdot\!\tilde s_i)=\pm1$ ⑤ solve α ⑥ $\tilde w=\sum\alpha_i\tilde s_i$ → $w$ + bias (last entry).
- Example 1 result: α=(−3.5, 0.75, 0.75), $w=(1,0)$, $b=−2$ → line $x_1=2$.
- Example 2 (kernel): map first, α=(−7, 4), $w=(1,1)$, $b=−3$.

**Non-linear SVM:** lift data to higher dimension where it's linearly separable. Avoids curse via large margin + **kernel trick** (implicit high-dim dot product).

**Kernels:** Linear $x_i\!\cdot\!x_j$; Polynomial $(x_i\!\cdot\!x_j+1)^d$; Gaussian $\exp(-\|x-y\|^2/2\sigma^2)$; RBF $\exp(-\gamma\|x_i-x_j\|^2)$. (Poly example: $x_i\!\cdot\!x_j=5$, $d=2$ → $36$.)

**Advantages:** high-dim accurate, kernels, memory efficient (SVs). **Disadvantages:** overfitting, no probabilities, slow on big data, bad if features > samples.

---

*✅ Study guide for Lecture 7 (Support Vector Machines) complete. **🎉 Now all 7 PPTs are done!***
