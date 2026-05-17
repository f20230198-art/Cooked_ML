# Lecture 7: Support Vector Machines (SVM)

> **Course:** BITS F464 — Machine Learning
> **Covers:** SVM intuition (max-margin), support vectors, the margin formula, the optimal-hyperplane optimization, a **full worked Linear SVM example**, Non-linear SVM (lifting, kernels), and a **full worked Kernel SVM example**.
>
> **How this chapter is written:** every idea opens with an everyday story, connects it to *why*, then the method/formula with symbols explained in the story's language. Both original worked examples kept; important formulas also get an **extra made-up example, fully solved**.

---

## Table of Contents

1. [SVM Introduction — the Max-Margin Idea](#1-svm-introduction--the-max-margin-idea)
2. [Margin & Support Vectors](#2-margin--support-vectors)
3. [The Margin Formula](#3-the-margin-formula)
4. [The Optimal Hyperplane (optimization)](#4-the-optimal-hyperplane-optimization-problem)
5. [Worked Example 1 — Linear SVM](#5-worked-example-1--linear-svm-full-trace)
6. [Non-Linear SVM — Lifting & Kernels](#6-non-linear-svm--lifting--kernels)
7. [Worked Example 2 — Kernel SVM](#7-worked-example-2--kernel-svm-full-trace)
8. [Advantages & Disadvantages](#8-advantages--disadvantages)
9. [The Night-Before Recap](#9-the-night-before-recap)

---

## 1. SVM Introduction — the Max-Margin Idea

### 🎯 Picture this

Two neighbours are arguing over where to put a fence between their yards. You *could* put it right up against one neighbour's flowerbed — but the first time a kid steps slightly over, there's a fight. The **safest** fence is the one drawn **smack in the middle, as far from both flowerbeds as possible**. Then small accidental encroachments from either side still don't cross it.

That "put the boundary as far as possible from the nearest things on both sides" is the entire SVM idea.

### 🔗 Why this matters

A linear discriminant: $g(x) = w^t x + w_0$ ($g>0$ → class 1, $g<0$ → class 2). **Many** lines can separate two classes — which is best?

> 🔑 **The SVM idea: choose the hyperplane AS FAR AS POSSIBLE from the closest examples** (maximize the distance to the nearest point).

**Why?** A boundary jammed next to a training point will misclassify a *new* point near it. A boundary far from all points → new points near old ones still land on the right side → better generalization.

```
  BAD (line close to points)        GOOD (line far from points)
   ●●● │ ■■■                          ●●●   │   ■■■
   ●●●│■■■   ← new point near          ●●●   │   ■■■
       small distance                   ← larger distance →
```

For the optimal hyperplane, distance to the closest **negative** = distance to the closest **positive** (it sits exactly in the middle — the fair fence).

---

## 2. Margin & Support Vectors

### 🎯 Picture this

Think of the fence as the centre of a **wide empty street** between the two yards. The **margin** is the street's full width. The **support vectors** are the few flowerpots sitting right on the kerb — they alone decide how wide the street can be and where it runs. Move a flowerpot deep inside a yard and nothing changes; move a kerb flowerpot and the whole street shifts.

### 🔗 Why this matters

> - **Margin** = twice the distance $b$ from the closest example to the separating hyperplane (the street's width).
> - **Support vectors** = the training points **closest to the hyperplane** (the kerb flowerpots — they "support" it).

```
   ●  ╱   b  ← margin boundary
   ● ╱─────── ← separating hyperplane
    ╱   b
   ╱ ■  ■     margin = 2b; circled kerb points = SUPPORT VECTORS
```

> 🔑 **Key insight:** only the **support vectors** matter. Delete every other point and the SVM boundary is *identical*. That's why SVM is **memory-efficient** — it only stores the support vectors.

> 💡 **SVM = Maximum Margin Classifier** — the widest possible street between the classes.

---

## 3. The Margin Formula

### 🎯 Picture this

To measure how far a flowerpot is from the fence, you need a *true* distance (in metres), not "5 ruler-marks of an arbitrary ruler." So we normalize the ruler. Once normalized, the street's half-width turns out to be exactly $1/\|w\|$ — so a **fatter street means a smaller $\|w\|$**.

### 🔗 Why this matters

**Absolute distance** of point $x$ from boundary $g(x)=0$:
$$\frac{|w^t x + w_0|}{\|w\|}$$

This is **scale-invariant**: scaling $g_1(x) = \alpha g(x)$ doesn't change it. For uniqueness we **normalize**: set $|w^t x_i + w_0| = 1$ for the closest example (a support vector). Then its distance to the boundary is $\dfrac{1}{\|w\|}$, and the **full margin** (both sides) is:

$$\boxed{\,m = \dfrac{2}{\|w\|}\,}$$

> 💡 **Crucial consequence:** maximizing the margin $\dfrac{2}{\|w\|}$ = **minimizing $\|w\|$**. Big $\|w\|$ → thin street; small $\|w\|$ → fat street.

#### ✏️ Practice example (made up — solved)

Two candidate boundaries separate the same data:
- Boundary A has $w = (3, 4)$, so $\|w\| = \sqrt{3^2+4^2} = \sqrt{25} = 5$ → margin $m = \dfrac{2}{5} = \mathbf{0.4}$
- Boundary B has $w = (1, 0)$, so $\|w\| = \sqrt{1^2+0^2} = 1$ → margin $m = \dfrac{2}{1} = \mathbf{2}$

→ Boundary B has the **wider street** (margin 2 > 0.4) because its $\|w\|$ is smaller. SVM would prefer **B** — confirming "minimize $\|w\|$ ⟺ maximize margin."

---

## 4. The Optimal Hyperplane (optimization problem)

### 🎯 Picture this

Goal: build the **widest possible street** such that **every flowerpot stays on its own side, outside the street** (no pot allowed in the empty street). "Widest street" + "everyone stays out of it on the correct side" — that's the whole optimization, in words.

### 🔗 Why this matters

**Maximize** $m = \dfrac{2}{\|w\|}$ **subject to** every point on its correct side, outside the margin:

$$w^t x_i + w_0 \geq +1 \;\text{ (positive)}, \qquad w^t x_i + w_0 \leq -1 \;\text{ (negative)}$$

With labels $z_i = +1$ (positive) / $-1$ (negative), both combine into $z_i(w^t x_i + w_0) \geq 1$.

Maximizing $\frac{2}{\|w\|}$ = minimizing $\frac12\|w\|^2$. So SVM training is:

$$\boxed{\;\text{minimize } J(w) = \tfrac{1}{2}\|w\|^2 \quad \text{s.t. } z_i(w^t x_i + w_0) \geq 1 \;\;\forall i\;}$$

> 🔑 A **constrained quadratic optimization** (solved with Lagrange multipliers → the α's below). The $\frac12$ and square just make the calculus clean; minimizing $\|w\|^2$ ⟺ minimizing $\|w\|$ ⟺ maximizing the margin.

---

## 5. Worked Example 1 — Linear SVM (full trace)

Follow the **6 steps** exactly.

**Given:** Positive (+): (3,1), (3,−1), (6,1), (6,−1). Negative (−): (1,0), (0,1), (0,−1), (−1,0). Find a simple separating SVM.

### Step 1 — Plot

```
 1 ■        ●        ●          ● = "+",  ■ = "−"
 0 ■   ■              (gap)
-1 ■        ●        ●
   -1 0 1   3        6   → x₁
```
"+" on the right, "−" on the left → a vertical-ish line separates them.

### Step 2 — Support vectors (by observation)

Points closest to the opposing class:
$$s_1 = \begin{pmatrix}1\\0\end{pmatrix} \text{(neg)}, \quad s_2 = \begin{pmatrix}3\\1\end{pmatrix}, \quad s_3 = \begin{pmatrix}3\\-1\end{pmatrix} \text{(pos)}$$

### Step 3 — Augment with a bias input of 1

Add a 3rd component = 1 (absorbs the bias $w_0$):
$$\tilde{s_1} = \begin{pmatrix}1\\0\\1\end{pmatrix}, \quad \tilde{s_2} = \begin{pmatrix}3\\1\\1\end{pmatrix}, \quad \tilde{s_3} = \begin{pmatrix}3\\-1\\1\end{pmatrix}$$

### Step 4–5 — Set up & solve the α equations

For each SV: $\sum_j \alpha_j (\tilde{s_j}\cdot \tilde{s_i}) = $ its label (−1 neg, +1 pos).

Dot products (e.g. $\tilde{s_1}\cdot\tilde{s_1} = 1+0+1 = 2$):

| Dot product | Value |
|---|---|
| $\tilde{s_1}\cdot\tilde{s_1}$ | $2$ |
| $\tilde{s_2}\cdot\tilde{s_1}$ | $4$ |
| $\tilde{s_3}\cdot\tilde{s_1}$ | $4$ |
| $\tilde{s_2}\cdot\tilde{s_2}$ | $11$ |
| $\tilde{s_3}\cdot\tilde{s_2}$ | $9$ |
| $\tilde{s_3}\cdot\tilde{s_3}$ | $11$ |

System:
$$2\alpha_1 + 4\alpha_2 + 4\alpha_3 = -1$$
$$4\alpha_1 + 11\alpha_2 + 9\alpha_3 = +1$$
$$4\alpha_1 + 9\alpha_2 + 11\alpha_3 = +1$$

Eqs 2 & 3 symmetric → $\alpha_2 = \alpha_3$. Sub into Eq 2: $4\alpha_1 + 20\alpha_2 = 1$; Eq 1: $2\alpha_1 + 8\alpha_2 = -1$.
$$\boxed{\alpha_1 = -3.5,\quad \alpha_2 = 0.75,\quad \alpha_3 = 0.75}$$

### Step 6 — Recover the hyperplane

$$\tilde{w} = \sum_i \alpha_i \tilde{s_i} = -3.5\begin{pmatrix}1\\0\\1\end{pmatrix} + 0.75\begin{pmatrix}3\\1\\1\end{pmatrix} + 0.75\begin{pmatrix}3\\-1\\1\end{pmatrix}$$
- 1st: $-3.5 + 2.25 + 2.25 = 1$
- 2nd: $0 + 0.75 - 0.75 = 0$
- 3rd: $-3.5 + 0.75 + 0.75 = -2$

$$\tilde{w} = \begin{pmatrix}1\\0\\-2\end{pmatrix} \;\Rightarrow\; \boxed{w = \begin{pmatrix}1\\0\end{pmatrix}, \quad b = -2}$$

**Hyperplane:** $1\cdot x_1 + 0\cdot x_2 - 2 = 0$ → **$x_1 = 2$** (vertical line at x=2).

✅ **Check:** (1,0): $1(1)-2 = -1 < 0$ → class −. ✓ (3,1): $1(3)-2 = +1 > 0$ → class +. ✓

> ✅ **Recipe:** ① plot ② pick SVs by eye ③ augment with "1" ④ build $\sum\alpha_j(\tilde s_j\!\cdot\!\tilde s_i)=\pm1$ ⑤ solve α ⑥ $\tilde w=\sum\alpha_i\tilde s_i$ → $w$ + bias (last entry).

#### ✏️ Practice example (made up — solved)

Just the **α-solving step** with 2 support vectors (simpler 2-equation system to practise the mechanics). Suppose after augmenting: $\tilde{s_1} = \begin{pmatrix}2\\0\\1\end{pmatrix}$ (negative, label −1), $\tilde{s_2} = \begin{pmatrix}4\\0\\1\end{pmatrix}$ (positive, label +1).

Dot products:
- $\tilde{s_1}\cdot\tilde{s_1} = 2(2)+0+1 = 5$
- $\tilde{s_2}\cdot\tilde{s_1} = 4(2)+0+1 = 9$
- $\tilde{s_2}\cdot\tilde{s_2} = 4(4)+0+1 = 17$

System ($\sum_j\alpha_j(\tilde s_j\cdot\tilde s_i)=$ label of $s_i$):
$$5\alpha_1 + 9\alpha_2 = -1 \quad (\text{for } s_1)$$
$$9\alpha_1 + 17\alpha_2 = +1 \quad (\text{for } s_2)$$

Solve: from Eq1, $\alpha_1 = \dfrac{-1 - 9\alpha_2}{5}$. Substitute into Eq2:
$$9\left(\frac{-1-9\alpha_2}{5}\right) + 17\alpha_2 = 1 \;\Rightarrow\; \frac{-9 - 81\alpha_2}{5} + 17\alpha_2 = 1$$
Multiply by 5: $-9 - 81\alpha_2 + 85\alpha_2 = 5 \;\Rightarrow\; -9 + 4\alpha_2 = 5 \;\Rightarrow\; 4\alpha_2 = 14 \;\Rightarrow\; \alpha_2 = 3.5$.
Then $\alpha_1 = \dfrac{-1 - 9(3.5)}{5} = \dfrac{-1 - 31.5}{5} = \dfrac{-32.5}{5} = -6.5$.

$$\boxed{\alpha_1 = -6.5,\quad \alpha_2 = 3.5}$$

Recover $\tilde w = \alpha_1\tilde s_1 + \alpha_2\tilde s_2 = -6.5\begin{pmatrix}2\\0\\1\end{pmatrix} + 3.5\begin{pmatrix}4\\0\\1\end{pmatrix}$:
- 1st: $-13 + 14 = 1$;  2nd: $0$;  3rd: $-6.5 + 3.5 = -3$

→ $w = (1, 0)$, $b = -3$ → hyperplane $x_1 = 3$ (the vertical line exactly **midway** between $x_1=2$ and $x_1=4$ — the fair fence, as expected).

---

## 6. Non-Linear SVM — Lifting & Kernels

### 🎯 Picture this

Imagine red and blue marbles on a **string** (1-D): blue in the middle, red on both ends. **No single cut** of the string separates them. Now **lift the string into a U-shape** (add a dimension): the red ends rise up, the blue middle stays low — and now a single **horizontal plane** slices cleanly between them. You didn't change the marbles; you changed the *space* they live in.

### 🔗 Why this matters

**Example:** 1-D points, "−" in the middle, "+" on both ends — not separable. Map up with $\Phi(x) = (x, x^2)$ → in 2-D they form a **parabola** a line *can* split.

```
  1-D (not separable):  ● ● ■ ■ ■ ■ ■ ● ●
                                ↓ Φ(x)=(x,x²)
  2-D (separable):       ●           ●   ← a straight line
                          ●__/■■■\__●        separates them
```

> 🔑 **The lifting trick:** project data into a **higher dimension** where it becomes linearly separable, then apply linear SVM there.

**The catch:** higher dimensions → **curse of dimensionality** (poor generalization + expensive). SVM dodges it via:
1. **Large-margin enforcement** → good generalization even in high dims.
2. **Kernel functions** → the high-dim computation is done **implicitly** (never actually compute high-dim coordinates).

### Common kernels

| Kernel | Formula |
|---|---|
| **Linear** | $k(x_i,x_j) = x_i\cdot x_j$ |
| **Polynomial** | $k(x_i,x_j) = (x_i\cdot x_j + 1)^d$ |
| **Gaussian** | $k(x,y) = \exp\left(-\dfrac{\|x-y\|^2}{2\sigma^2}\right)$ |
| **RBF** | $k(x_i,x_j) = \exp\left(-\gamma\|x_i-x_j\|^2\right)$ |

> 💡 **Kernel trick intuition:** a kernel computes the *dot product in the high-dim space* using only the *original* vectors — the power of high dimensions without ever building them. Gaussian/RBF are the common defaults.

#### 🧠 Worked example — polynomial kernel

$x_i=(1,2)$, $x_j=(3,1)$, degree $d=2$:
$$x_i\cdot x_j = 1(3)+2(1) = 5, \qquad k = (5+1)^2 = 6^2 = \mathbf{36}$$

#### ✏️ Practice example (made up — solved)

**Polynomial kernel**, $x_i = (2, 0)$, $x_j = (1, 3)$, degree $d = 3$:
$$x_i\cdot x_j = 2(1) + 0(3) = 2, \qquad k = (2 + 1)^3 = 3^3 = \mathbf{27}$$

**Gaussian kernel** on the same points with $\sigma = 2$. First $\|x_i - x_j\|^2$: difference $= (2-1,\ 0-3) = (1, -3)$, so $\|x_i-x_j\|^2 = 1^2 + (-3)^2 = 1 + 9 = 10$.
$$k = \exp\left(-\frac{10}{2(2)^2}\right) = \exp\left(-\frac{10}{8}\right) = \exp(-1.25) \approx \mathbf{0.287}$$
→ Both kernels return a single number computed **only from the original 2-D vectors** — no high-dimensional coordinates were ever built. That's the kernel trick.

---

## 7. Worked Example 2 — Kernel SVM (full trace)

Same 6-step procedure, with a **mapping applied first**.

**Given:** Positive (+): (2,2), (2,−2), (−2,−2), (−2,2) (outer corners). Negative (−): (1,1), (1,−1), (−1,−1), (−1,1) (inner corners).

```
 2 ●       ●        ● = "+" (outer),  ■ = "−" (inner)
 1   ■   ■          NOT linearly separable — "−" is
-1   ■   ■            surrounded by "+" on all sides
-2 ●       ●
   -2 -1 0 1 2 → x₁
```

### Step 2 — Apply the mapping

$$\Phi_1\begin{pmatrix}x_1\\x_2\end{pmatrix} = \begin{cases} \begin{pmatrix}4-x_2+|x_1-x_2|\\ 4-x_1+|x_1-x_2|\end{pmatrix} & \text{if } \sqrt{x_1^2+x_2^2} > 2 \\[3mm] \begin{pmatrix}x_1\\x_2\end{pmatrix} & \text{otherwise} \end{cases}$$

Outer points (radius > 2) transform; inner points stay:
- **Positive:** (2,2)→(2,2), (2,−2)→(10,6), (−2,−2)→(6,6), (−2,2)→(6,10)
- **Negative:** stay (1,1), (1,−1), (−1,−1), (−1,1)

Now linearly separable.

### Step 3 — Support vectors

$$s_1 = \begin{pmatrix}1\\1\end{pmatrix}\text{(neg)}, \quad s_2 = \begin{pmatrix}2\\2\end{pmatrix}\text{(pos)}$$

### Step 4 — Augment with bias 1

$$\tilde{s_1} = \begin{pmatrix}1\\1\\1\end{pmatrix}, \quad \tilde{s_2} = \begin{pmatrix}2\\2\\1\end{pmatrix}$$

### Step 5 — Solve the α equations

Dot products: $\tilde{s_1}\cdot\tilde{s_1} = 3$, $\tilde{s_2}\cdot\tilde{s_1} = 5$, $\tilde{s_2}\cdot\tilde{s_2} = 9$.
$$3\alpha_1 + 5\alpha_2 = -1$$
$$5\alpha_1 + 9\alpha_2 = +1$$
From Eq1 $\alpha_1 = \frac{-1-5\alpha_2}{3}$. Substitute → $\boxed{\alpha_1 = -7,\quad \alpha_2 = 4}$

### Step 6 — Recover the hyperplane

$$\tilde{w} = -7\begin{pmatrix}1\\1\\1\end{pmatrix} + 4\begin{pmatrix}2\\2\\1\end{pmatrix}$$
- 1st: $-7 + 8 = 1$;  2nd: $-7 + 8 = 1$;  3rd: $-7 + 4 = -3$

$$\tilde{w} = \begin{pmatrix}1\\1\\-3\end{pmatrix} \;\Rightarrow\; \boxed{w = \begin{pmatrix}1\\1\end{pmatrix}, \quad b = -3}$$

**Hyperplane:** $x_1 + x_2 - 3 = 0$ in the *mapped* space — a **non-linear** boundary (a curve) back in the original space, separating inner from outer.

> 🔑 **The big idea:** the *same linear SVM procedure*, just preceded by a mapping/kernel. Linear in the lifted space = non-linear in the original space. That's how SVM handles non-separable data (the lifted-string trick).

---

## 8. Advantages & Disadvantages

| ✅ Advantages | ❌ Disadvantages |
|---|---|
| Accurate in **high-dimensional** space | Prone to **overfitting** (wrong kernel) |
| **Different kernels** for different problems | **No probability estimate** (just the class) |
| **Memory efficient** — only stores support vectors | Efficient only for **small/average** datasets |
| Strong generalization (max margin) | **Bad if features > samples** |

> 🔑 **One-liners:** SVM = max-margin classifier defined only by support vectors. Strength = high-dim accuracy + kernels. Weakness = slow on big data, no probabilities, sensitive to kernel choice.

---

## 9. The Night-Before Recap

**SVM idea** (the fair fence): pick the separating hyperplane with the **maximum margin** (farthest from the closest points) → best generalization.

**Support vectors** = the kerb points closest to the boundary; they alone define it (delete the rest → same line → memory-efficient).

**Margin** $m = \dfrac{2}{\|w\|}$ — the street width. Maximize margin ⟺ **minimize $\|w\|$** ⟺ minimize $\frac12\|w\|^2$. (Practice: $\|w\|=5\Rightarrow m=0.4$; $\|w\|=1\Rightarrow m=2$, so smaller $\|w\|$ wins.)

**Optimization:** minimize $\frac12\|w\|^2$ s.t. $z_i(w^tx_i+w_0)\geq1$ (z=+1 pos, −1 neg) — widest street, everyone outside it on the correct side.

**6-step procedure:** ① plot ② pick SVs ③ augment with 1 ④ build $\sum\alpha_j(\tilde s_j\!\cdot\!\tilde s_i)=\pm1$ ⑤ solve α ⑥ $\tilde w=\sum\alpha_i\tilde s_i$ → $w$ + bias (last entry).
- Example 1: α=(−3.5, 0.75, 0.75), $w=(1,0)$, $b=−2$ → line $x_1=2$.
- Practice (2-SV): α=(−6.5, 3.5) → $w=(1,0)$, $b=−3$ → line $x_1=3$ (midway between SVs).
- Example 2 (kernel): map first, α=(−7, 4), $w=(1,1)$, $b=−3$.

**Non-linear SVM** = lift the string into a U: project to a higher dimension where it's linearly separable. Avoids the curse via large margin + **kernel trick** (implicit high-dim dot product).

**Kernels:** Linear $x_i\!\cdot\!x_j$; Polynomial $(x_i\!\cdot\!x_j+1)^d$; Gaussian $\exp(-\|x-y\|^2/2\sigma^2)$; RBF $\exp(-\gamma\|x_i-x_j\|^2)$.
- Worked poly: $x_i\!\cdot\!x_j=5$, $d=2$ → $36$. Practice poly: $x_i\!\cdot\!x_j=2$, $d=3$ → $27$; Gaussian ($\sigma=2$, dist²=10) → ≈0.287.

**Advantages:** high-dim accurate, kernels, memory efficient (SVs). **Disadvantages:** overfitting, no probabilities, slow on big data, bad if features > samples.

---

*✅ Study guide for Lecture 7 (Support Vector Machines) complete. 🎉 All 7 chapters rewritten.*
