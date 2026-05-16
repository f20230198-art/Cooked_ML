# Lecture 3: Discriminant Functions

> **Course:** BITS F464 — Machine Learning
> **Covers:** Classification basics (decision regions/boundaries), generalized linear model, two-class & multi-class discriminant functions, One-vs-Rest & One-vs-One, the geometry (perpendicular distance), Least Squares for classification + its problems, and Fisher Linear Discriminant / LDA (within-class & between-class scatter matrices).
>
> **How to read this:** Every formula has a worked numerical example. The geometry of the two-class discriminant and the LDA scatter matrices are the exam-likely problems — both are worked end-to-end with numbers.

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [The Generalized Linear Model](#2-the-generalized-linear-model)
3. [Two-Class Discriminant Function (+ geometry)](#3-two-class-discriminant-function--the-geometry)
4. [Multiple Classes](#4-multiple-classes)
5. [K Linear Functions Approach](#5-the-k-linear-functions-approach)
6. [Least Squares for Classification](#6-least-squares-for-classification)
7. [Problems with Least Squares](#7-problems-with-least-squares)
8. [Fisher Linear Discriminant / LDA](#8-fisher-linear-discriminant--lda)
9. [Quick Revision Sheet](#9-quick-revision-sheet)

---

## 1. What is Classification?

> **Goal of classification:** take an input vector $\mathbf{x}$ and assign it to **one of $K$ discrete classes** $C_k$ (where $k = 1, \dots, K$).

This is different from regression (which predicts a continuous *number*). Classification predicts a *category*.

**Key vocabulary:**

| Term | Meaning |
|---|---|
| **Decision region** | A chunk of input space that all maps to one class |
| **Decision boundary** (or **decision surface**) | The dividing line/surface *between* decision regions |
| **Linear model for classification** | The decision boundary is a **linear function** of $\mathbf{x}$ (a straight line/flat plane) |

```
  (x)₂ │  R  R  R ╲ "No"
       │ R R R R  ╲          ← straight decision boundary
       │  R R R    ╲   B B
       │ R R R      ╲ B B B  "yes"
       │   region R₂  ╲ B B B   region R₁
       └───────────────╲─────────► (x)₁
```

> 🔑 **Hyperplane fact (exam favourite):** In a $D$-dimensional input space, a linear decision boundary is a **$(D-1)$-dimensional hyperplane**.
> - In 2D (a plane), the boundary is a **1D line**.
> - In 3D (a cube), the boundary is a **2D plane**.
> - General rule: boundary dimension = input dimension − 1.

**Linearly separable** = the classes can be separated **exactly** by a linear decision surface (no points on the wrong side).

---

## 2. The Generalized Linear Model

The general model for classification:

$$y(\mathbf{x}) = f\big(\mathbf{w}^T\mathbf{x} + w_0\big)$$

| Symbol | Meaning |
|---|---|
| $\mathbf{w}$ | weight vector (defines the boundary's orientation) |
| $w_0$ | bias / offset |
| $\mathbf{w}^T\mathbf{x} + w_0$ | a linear function of the input (a number) |
| $f(\cdot)$ | a fixed **non-linear activation function** that squashes the number into a class decision |

**Example activation function (step function):**
$$f(u) = \begin{cases} 1 & \text{if } u \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

> 💡 **Why apply $f$?** $\mathbf{w}^T\mathbf{x}+w_0$ is just a real number (could be −7.3 or +42). We need a *class label*. The step function turns "is it positive?" into a clean 1 or 0. Even though $f$ is non-linear, **the decision boundary itself is still linear in $\mathbf{x}$** (because the boundary is where $\mathbf{w}^T\mathbf{x}+w_0=0$, a linear equation).

#### 🧠 Worked example

Let $\mathbf{w} = (2, -1)$, $w_0 = -1$, and test point $\mathbf{x} = (3, 1)$:
$$\mathbf{w}^T\mathbf{x} + w_0 = (2)(3) + (-1)(1) + (-1) = 6 - 1 - 1 = 4$$
Since $4 \geq 0$, $f(4) = 1$ → assign $\mathbf{x}$ to **class 1**.

---

## 3. Two-Class Discriminant Function + the Geometry

For a **2-class** problem, the simple linear discriminant function is:

$$y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0$$

**The decision rule:**

| Condition | Decision |
|---|---|
| $y(\mathbf{x}) \geq 0$ | class $C_1$ |
| $y(\mathbf{x}) < 0$ | class $C_2$ |
| $y(\mathbf{x}) = 0$ | exactly **on the decision boundary** |

- The **threshold is the negative of the bias** ($-w_0$): you're really checking whether $\mathbf{w}^T\mathbf{x} \geq -w_0$.

### The geometry (slide 5 diagram — important for exam)

```
        x₂ │ y>0
           │ y=0  ╲  R₁           • x  (a test point)
           │ y<0 ╲ R₂           ╱┊
           │      ╲     ╲    ╱  ┊ ← y(x)/‖w‖  (signed distance
           │   w ↗ ╲      ╲ ╱   ┊    of x from boundary)
           │      ╱ ╲   x⊥•     ┊
           │    ╱     ╲    ╲     │
           └──────────────╲──────► x₁
                   −w₀/‖w‖ (distance of boundary from origin)
```

**Two distance facts to remember:**

1. **Perpendicular (signed) distance of a point $\mathbf{x}$ from the boundary:**
$$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|} = \frac{\mathbf{w}^T\mathbf{x} + w_0}{\|\mathbf{w}\|}$$
   (The slide writes the projection of $\mathbf{x}$ in the $\mathbf{w}$ direction as $\dfrac{\mathbf{w}^T\mathbf{x}}{\|\mathbf{w}\|}$.)

2. **Distance of the boundary itself from the origin** $= \dfrac{-w_0}{\|\mathbf{w}\|}$.

Here $\|\mathbf{w}\|$ ("norm of w") = the length of the weight vector $= \sqrt{w_1^2 + w_2^2 + \dots}$.

> 💡 **Intuition:** $\mathbf{w}$ points perpendicular to the boundary (it's the "uphill" direction of $y$). The bigger $y(\mathbf{x})$, the further $\mathbf{x}$ sits from the boundary on the $C_1$ side. Dividing by $\|\mathbf{w}\|$ converts the raw score into an actual geometric distance.

#### 🧠 Worked example — classify a point AND find its distance

Let $\mathbf{w} = (3, 4)$, $w_0 = -10$, test point $\mathbf{x} = (2, 2)$.

**Step 1 — compute $y(\mathbf{x})$:**
$$y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0 = (3)(2) + (4)(2) + (-10) = 6 + 8 - 10 = 4$$

**Step 2 — classify:** $y = 4 \geq 0$ → **class $C_1$**.

**Step 3 — length of $\mathbf{w}$:**
$$\|\mathbf{w}\| = \sqrt{3^2 + 4^2} = \sqrt{9+16} = \sqrt{25} = 5$$

**Step 4 — perpendicular distance from the boundary:**
$$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|} = \frac{4}{5} = \mathbf{0.8}$$

**Step 5 — distance of the boundary from origin:**
$$\frac{-w_0}{\|\mathbf{w}\|} = \frac{-(-10)}{5} = \frac{10}{5} = \mathbf{2}$$

> ✅ So point (2,2) is class $C_1$, sits **0.8 units** from the boundary, and the boundary is **2 units** from the origin. This exact style of problem is highly exam-likely.

---

## 4. Multiple Classes

A single linear discriminant separates only **two** classes (one hyperplane). For $K > 2$ classes, we combine binary classifiers. **Two strategies:**

| Method | How many classifiers | What each does |
|---|---|---|
| **One-vs-the-Rest** (OvR) | $K - 1$ classifiers | Each separates one class $C_k$ from **all others** |
| **One-vs-One** (OvO) | $\dfrac{K(K-1)}{2}$ classifiers | One for **every pair** of classes |

#### 🧠 Worked example — count classifiers for K = 4 classes

- **One-vs-Rest:** $K - 1 = 4 - 1 = \mathbf{3}$ classifiers.
- **One-vs-One:** $\dfrac{K(K-1)}{2} = \dfrac{4 \times 3}{2} = \dfrac{12}{2} = \mathbf{6}$ classifiers.

> ⚠️ **The ambiguity problem (the green "?" region in the slide):** both OvR and OvO can produce regions where the votes **conflict** — a point could be claimed by 2 classes or by none. This is the main weakness of naïvely combining binary classifiers.

---

## 5. The K Linear Functions Approach

A cleaner fix: build **$K$ linear functions**, one per class:

$$y_k(\mathbf{x}) = \mathbf{w}_k^T\mathbf{x} + w_{k0}$$

**Decision rule:** assign $\mathbf{x}$ to class $C_k$ if
$$y_k(\mathbf{x}) > y_j(\mathbf{x}) \quad \text{for all } j \neq k$$
(i.e., **pick the class whose function gives the biggest score** — a "winner takes all").

**Decision boundary between class $C_k$ and $C_j$** is where their scores tie, $y_k(\mathbf{x}) = y_j(\mathbf{x})$, giving the $(D-1)$-dim hyperplane:

$$(\mathbf{w}_k - \mathbf{w}_j)^T\mathbf{x} + (w_{k0} - w_{j0}) = 0$$

> 🔑 **Key property:** the decision regions of this discriminant are always **singly connected and convex** — no fragmented or ambiguous regions like OvR/OvO. (Convex = the line between any two points of a region stays inside that region.)

#### 🧠 Worked example — winner-takes-all

3 classes with $y_1(\mathbf{x}) = 2$, $y_2(\mathbf{x}) = 5$, $y_3(\mathbf{x}) = 3$ for some point $\mathbf{x}$.
→ Largest is $y_2 = 5$ → assign $\mathbf{x}$ to **class $C_2$**.

---

## 6. Least Squares for Classification

**Question:** how do we actually *learn* the parameters $\mathbf{w}_k, w_{k0}$? **One approach: least squares** (same idea as in regression — minimize squared error).

We use **1-of-K (one-hot) target vectors**: e.g. for 3 classes, class 2 → $\mathbf{t} = (0, 1, 0)$.

The error to minimize over **all $N$ examples** and **all $K$ class-components**:

$$E(\mathbf{W}) = \frac{1}{2}\sum_{n=1}^{N}\sum_{k=1}^{K}\big(y_k(\mathbf{x}_n) - t_{nk}\big)^2$$

| Symbol | Meaning |
|---|---|
| $N$ | number of training examples |
| $K$ | number of classes |
| $y_k(\mathbf{x}_n)$ | model's score for class $k$ on example $n$ |
| $t_{nk}$ | true target (1 if example $n$ is class $k$, else 0) |
| $\frac{1}{2}$ | constant that makes the calculus cleaner when differentiating |

> 💡 This is just MSE applied to classification: for every example, for every class slot, square the difference between predicted score and the 0/1 target, add them all up, find the $\mathbf{W}$ that makes it smallest.

#### 🧠 Worked example — compute the error

One example $n$, 3 classes. True class = 2, so target $\mathbf{t} = (0, 1, 0)$. Model outputs $\mathbf{y} = (0.2, 0.7, 0.3)$.

$$E = \frac{1}{2}\Big[(0.2-0)^2 + (0.7-1)^2 + (0.3-0)^2\Big]$$
$$= \frac{1}{2}\big[0.04 + 0.09 + 0.09\big] = \frac{1}{2}(0.22) = \mathbf{0.11}$$

(Sum this over all $N$ examples for the total error.)

---

## 7. Problems with Least Squares

Least squares *looks* fine for clean data, but has serious flaws for classification:

### Problem 1 — It's not robust to outliers / "easy" points (slide 9)

- Adding **extra points that are very easy to classify** (far from the boundary) actually makes the boundary **worse**!
- **Why?** If the target is 1, but a far-away point produces a huge score like 10, then $(10 - 1)^2 = 81$ is a **massive squared error**. Least squares panics over this huge error and **rotates the boundary** to reduce it — even though that point was already classified correctly.

```
   Before adding far points:        After adding far points:
   × ×  ╲  good boundary            × ×  ╲  boundary got DRAGGED
   ×× × ╲ o o                       ×× ×  ╲  toward the far cluster
   × ×   ╲ o o o                    × ×    ╲ o o   ← misclassifies now!
          ╲                                ╲  o
              o (far easy point)      far point o o (pulls line)
```

> 🔑 **Root cause:** least squares penalizes being "too correct" (large positive score) just as much as being wrong, because it squares *all* deviations from the target. A point far on the correct side shouldn't be a problem — but to least squares, its big error looks like a disaster.

### Problem 2 — It masks classes in multi-class problems (slide 10)

With 3+ classes, the least-squares boundaries can **completely squeeze out the middle class** (green class gets almost no region — it's "masked" by red and blue). A better method (like logistic regression / LDA) gives the green class its proper region.

> ✅ **Takeaway:** Least squares for classification is simple but **fragile** — sensitive to outliers and prone to masking. Motivates better methods (logistic regression, LDA).

---

## 8. Fisher Linear Discriminant / LDA

> **LDA (Linear Discriminant Analysis)** finds a **projection direction** that **best separates the classes**. It's also used for **dimensionality reduction** (as a preprocessing step).

### 8.1 The core idea

We want to **project** high-dimensional points down onto a line (a direction), then separate classes on that line. **Which direction is best?**

- ❌ **Naïve idea:** project onto the line connecting the two class **means**. Problem: if classes are stretched/spread along that direction, the projections **overlap badly**.
- ✅ **Fisher's criterion:** choose the direction that **maximizes the ratio:**

$$\text{maximize} \quad \frac{\text{between-class separation (inter-class)}}{\text{within-class variance (intra-class)}}$$

```
  BAD separability                 GOOD separability
  (project onto x₁ axis)           (Fisher's projection)
   ■ ■■■                              ■■■
   ■■● ●■  ← classes overlap          ■■■   ← clean gap
   ●●●●●     when projected           ━━━━
   ──────────► projection             ●●●
                                      ●●●●  ← classes well apart
```

> 💡 **Plain English:** Push the **class centres far apart** (big numerator) while keeping **each class tightly clustered** (small denominator). Maximize "spread *between* groups ÷ spread *within* groups."

### 8.2 The setup

Assume $C$ classes. Class $i$ has $M_i$ samples. Total samples:
$$M = \sum_{i=1}^{C} M_i$$

- $\boldsymbol{\mu}_i$ = mean of class $i$.
- $\boldsymbol{\mu}$ = mean of the whole dataset: $\displaystyle\boldsymbol{\mu} = \frac{1}{C}\sum_{i=1}^{C}\boldsymbol{\mu}_i$

### 8.3 Within-class scatter matrix $S_w$

Measures **how spread out each class is around its own mean** (we want this **small**):

$$S_w = \sum_{i=1}^{C}\sum_{j=1}^{M_i}(\mathbf{x}_{ij} - \boldsymbol{\mu}_i)(\mathbf{x}_{ij} - \boldsymbol{\mu}_i)^T$$

Read it as: for every class $i$, for every point $j$ in it, take (point − its class mean), and accumulate. **Small $S_w$ = tight, compact classes.**

### 8.4 Between-class scatter matrix $S_b$

Measures **how far the class means are from the overall mean** (we want this **large**):

$$S_b = \sum_{i=1}^{C}(\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T$$

**Large $S_b$ = class centres well separated.**

> 🔑 **LDA's whole job:** find direction $\mathbf{w}$ that **maximizes $\dfrac{S_b}{S_w}$** → classes far apart *and* tightly packed.

#### 🧠 Worked example — compute means and scatter (1-D, 2 classes)

Class 1 points: {1, 2, 3}. Class 2 points: {6, 7, 8}.

**Step 1 — class means:**
$$\mu_1 = \frac{1+2+3}{3} = 2, \qquad \mu_2 = \frac{6+7+8}{3} = 7$$

**Step 2 — overall mean** (mean of class means, per slide's formula):
$$\mu = \frac{\mu_1 + \mu_2}{2} = \frac{2 + 7}{2} = 4.5$$

**Step 3 — within-class scatter $S_w$** = sum of (point − class mean)² over all points:
- Class 1: $(1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2$
- Class 2: $(6-7)^2 + (7-7)^2 + (8-7)^2 = 1 + 0 + 1 = 2$
- $S_w = 2 + 2 = \mathbf{4}$ (small → classes are tight ✓)

**Step 4 — between-class scatter $S_b$** = sum of (class mean − overall mean)²:
$$S_b = (\mu_1 - \mu)^2 + (\mu_2 - \mu)^2 = (2-4.5)^2 + (7-4.5)^2 = 6.25 + 6.25 = \mathbf{12.5}$$

**Step 5 — Fisher ratio:**
$$\frac{S_b}{S_w} = \frac{12.5}{4} = \mathbf{3.125}$$

> A **high ratio (3.125)** means these two classes are **well separated relative to their internal spread** — exactly what LDA tries to maximize. If the classes overlapped, $S_w$ would be large and the ratio small.

---

## 9. Quick Revision Sheet

> Night-before-exam cheat sheet.

**Classification** = assign $\mathbf{x}$ to one of $K$ classes. **Decision boundary** separates **decision regions**. In $D$ dims, boundary is a **$(D-1)$-dim hyperplane**.

**Generalized linear model:** $y(\mathbf{x}) = f(\mathbf{w}^T\mathbf{x} + w_0)$, $f$ = activation (e.g. step). Boundary still **linear in x**.

**Two-class discriminant:** $y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0$.
- $y \geq 0$ → $C_1$; $y < 0$ → $C_2$; $y = 0$ → boundary. Threshold $= -w_0$.
- Perpendicular distance of point: $r = \dfrac{y(\mathbf{x})}{\|\mathbf{w}\|}$.
- Boundary distance from origin: $\dfrac{-w_0}{\|\mathbf{w}\|}$. ($\|\mathbf{w}\|=\sqrt{w_1^2+w_2^2+\dots}$)

**Multi-class:**

| Method | # classifiers (K classes) |
|---|---|
| One-vs-Rest | $K-1$ |
| One-vs-One | $K(K-1)/2$ |
| K linear funcs | $K$ — pick max $y_k$; regions **convex & singly connected** |

Boundary between $C_k,C_j$: $(\mathbf{w}_k-\mathbf{w}_j)^T\mathbf{x}+(w_{k0}-w_{j0})=0$.

**Least squares for classification:** minimize $E(\mathbf{W})=\frac12\sum_n\sum_k (y_k(\mathbf{x}_n)-t_{nk})^2$.
- ❌ Not robust: far "easy" points create huge squared errors → boundary gets dragged.
- ❌ Masks the middle class in multi-class problems.

**LDA (Fisher):** maximize $\dfrac{S_b}{S_w}$ = $\dfrac{\text{between-class separation}}{\text{within-class variance}}$. Also used for dimensionality reduction.
- Within-class: $S_w = \sum_i\sum_j (\mathbf{x}_{ij}-\boldsymbol{\mu}_i)(\mathbf{x}_{ij}-\boldsymbol{\mu}_i)^T$ — want **small** (tight classes).
- Between-class: $S_b = \sum_i (\boldsymbol{\mu}_i-\boldsymbol{\mu})(\boldsymbol{\mu}_i-\boldsymbol{\mu})^T$ — want **large** (separated centres).
- Worked example: classes {1,2,3} & {6,7,8} → $S_w=4$, $S_b=12.5$, ratio $=3.125$ (well separated).

---

*✅ Study guide for Lecture 3 (Discriminant Functions) complete. Send the next PPT whenever ready.*
