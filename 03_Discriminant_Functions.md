# Lecture 3: Discriminant Functions

> **Course:** BITS F464 — Machine Learning
> **Covers:** Classification basics (decision regions/boundaries), the generalized linear model, two-class & multi-class discriminant functions, One-vs-Rest & One-vs-One, the geometry (perpendicular distance), Least Squares for classification + its problems, and Fisher Linear Discriminant / LDA.
>
> **How this chapter is written:** every idea opens with an everyday story, connects it to *why* we do the thing, then the method/formula with symbols explained in the story's language. All original worked examples kept; important formulas also get an **extra made-up example, fully solved**.

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
9. [The Night-Before Recap](#9-the-night-before-recap)

---

## 1. What is Classification?

### 🎯 Picture this

Imagine sorting incoming mail into **bins**: "bills", "personal", "junk". Every letter goes into exactly one bin based on its features. The **invisible rules** that decide which bin — that's classification. And there are imaginary **lines on the table** separating where "bills" letters pile up from where "junk" piles up.

### 🔗 Why this matters

> **Goal of classification:** take an input $\mathbf{x}$ and assign it to **one of $K$ discrete classes** $C_k$.

Different from regression (which predicts a *number*). Classification predicts a *category* (which bin).

| Term | Meaning |
|---|---|
| **Decision region** | A chunk of input space that all maps to one class (where one bin's letters pile up) |
| **Decision boundary** | The dividing line/surface *between* regions (the line on the table) |
| **Linear model for classification** | The boundary is a **straight line / flat plane** |

```
  (x)₂ │  R  R  R ╲ "No"
       │ R R R R  ╲          ← straight decision boundary
       │  R R R    ╲   B B
       │ R R R      ╲ B B B  "yes"
       │   region R₂  ╲ B B B   region R₁
       └───────────────╲─────────► (x)₁
```

> 🔑 **Hyperplane fact:** in a $D$-dimensional input space, a linear boundary is a **$(D-1)$-dimensional hyperplane**. In 2D the boundary is a 1D **line**; in 3D it's a 2D **plane**. Rule: boundary dimension = input dimension − 1.

**Linearly separable** = the classes can be split **exactly** by one linear surface (no point on the wrong side).

---

## 2. The Generalized Linear Model

### 🎯 Picture this

A bouncer at a club computes a "score" from your features (age, dress code, guest-list status), each weighted by importance. Then a hard rule converts the score into a yes/no: **score ≥ 0 → you're in; otherwise → out.** The score is a smooth number; the *decision* is a clean category.

### 🔗 Why this matters

$$y(\mathbf{x}) = f(\mathbf{w}^T\mathbf{x} + w_0)$$

| Symbol | Meaning |
|---|---|
| $\mathbf{w}$ | weight vector (sets the boundary's orientation) |
| $w_0$ | bias / offset |
| $\mathbf{w}^T\mathbf{x} + w_0$ | a linear score (just a number) |
| $f(\cdot)$ | an **activation function** that squashes the score into a class |

**Step activation:** $f(u) = 1$ if $u \geq 0$, else $0$.

> 💡 **Why apply $f$?** $\mathbf{w}^T\mathbf{x}+w_0$ is just a number (could be −7.3 or +42). We need a *class label*. The step function turns "is it positive?" into a clean 1/0 (the bouncer's yes/no). Even though $f$ is non-linear, the **boundary is still linear in $\mathbf{x}$** — it's where $\mathbf{w}^T\mathbf{x}+w_0=0$, a linear equation.

#### 🧠 Worked example

$\mathbf{w} = (2, -1)$, $w_0 = -1$, test point $\mathbf{x} = (3, 1)$:
$$\mathbf{w}^T\mathbf{x} + w_0 = (2)(3) + (-1)(1) + (-1) = 6 - 1 - 1 = 4$$
$4 \geq 0$ → $f(4) = 1$ → **class 1**.

#### ✏️ Practice example (made up — solved)

A loan-approval bouncer: $\mathbf{w} = (3, 2)$ on features (credit score scaled, income scaled), $w_0 = -10$.

*Applicant A* $\mathbf{x} = (1, 2)$: $\;(3)(1)+(2)(2)-10 = 3+4-10 = -3$. $-3 < 0$ → $f = 0$ → **rejected**.
*Applicant B* $\mathbf{x} = (2, 3)$: $\;(3)(2)+(2)(3)-10 = 6+6-10 = +2$. $2 \geq 0$ → $f = 1$ → **approved**.

→ Same weights, the score crossing 0 is the entire decision — exactly the bouncer flipping yes/no at the threshold.

---

## 3. Two-Class Discriminant Function + the Geometry

> 🎯 **The whole idea in one picture:** draw a straight line through the data — one side is class 1, the other side class 2. The function $y(\mathbf{x})$ is a **score**: its *sign* says which side you're on, its *size* says *how far* from the line. The "geometry" formulas just convert that score into an actual distance (cm/units).

For a 2-class problem:

$$y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0$$

| Condition | Decision |
|---|---|
| $y(\mathbf{x}) \geq 0$ | class $C_1$ |
| $y(\mathbf{x}) < 0$ | class $C_2$ |
| $y(\mathbf{x}) = 0$ | exactly **on the boundary** |

The threshold is $-w_0$ (you're really checking $\mathbf{w}^T\mathbf{x} \geq -w_0$).

### The geometry

```
        x₂ │ y>0
           │ y=0  ╲  R₁           • x  (a test point)
           │ y<0 ╲ R₂           ╱┊
           │      ╲     ╲    ╱  ┊ ← y(x)/‖w‖  (signed distance)
           │   w ↗ ╲      ╲ ╱   ┊
           │      ╱ ╲   x⊥•     ┊
           └──────────────╲──────► x₁
                   −w₀/‖w‖ (boundary's distance from origin)
```

> 💡 **Why divide by $\|\mathbf{w}\|$?** The raw score depends on how "big" you happened to make $\mathbf{w}$ — double $\mathbf{w}$ and every score doubles even though the line didn't move. Dividing by $\|\mathbf{w}\|$ (the *length* of $\mathbf{w}$) cancels that scaling and gives a **true distance**. Like converting "5 ruler-marks" into "5 cm" so the number actually means something.

**Two distance facts:**

1. **Perpendicular (signed) distance of point $\mathbf{x}$ from the boundary:**
$$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|} = \frac{\mathbf{w}^T\mathbf{x} + w_0}{\|\mathbf{w}\|}$$

2. **Distance of the boundary from the origin** $= \dfrac{-w_0}{\|\mathbf{w}\|}$.

Here $\|\mathbf{w}\| = \sqrt{w_1^2 + w_2^2 + \dots}$ (Pythagoras on the weights).

#### 🧠 Worked example — classify a point AND find its distance

$\mathbf{w} = (3, 4)$, $w_0 = -10$, test point $\mathbf{x} = (2, 2)$.

1. $y(\mathbf{x}) = (3)(2) + (4)(2) + (-10) = 6 + 8 - 10 = 4$
2. $y = 4 \geq 0$ → **class $C_1$**
3. $\|\mathbf{w}\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$
4. perpendicular distance $r = \frac{4}{5} = \mathbf{0.8}$
5. boundary from origin $= \frac{-(-10)}{5} = \frac{10}{5} = \mathbf{2}$

#### ✏️ Practice example (made up — solved)

$\mathbf{w} = (1, 2)$, $w_0 = -4$, test point $\mathbf{x} = (4, 3)$.

1. $y(\mathbf{x}) = (1)(4) + (2)(3) + (-4) = 4 + 6 - 4 = 6$
2. $6 \geq 0$ → **class $C_1$**
3. $\|\mathbf{w}\| = \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.236$
4. perpendicular distance $r = \frac{6}{2.236} \approx \mathbf{2.68}$ units from the boundary (well clear of it, confidently $C_1$)
5. boundary from origin $= \frac{-(-4)}{2.236} = \frac{4}{2.236} \approx \mathbf{1.79}$

→ Compare to a point right on the line, e.g. $\mathbf{x}=(2,1)$: $y = 2+2-4 = 0$ → distance $0$ → sits exactly on the boundary (ambiguous).

---

## 4. Multiple Classes

### 🎯 Picture this

One referee can only judge a **two-person** race ("who crossed first"). For a **4-person** race you need a scheme. Two options: (a) each referee judges "runner X vs everyone else" → a few referees; (b) one referee per **pair** of runners → many referees, then tally.

### 🔗 Why this matters

A single linear discriminant separates only **two** classes. For $K > 2$, combine binary classifiers:

| Method | # classifiers | What each does |
|---|---|---|
| **One-vs-Rest** (OvR) | $K - 1$ | Each separates one class from **all others** |
| **One-vs-One** (OvO) | $\dfrac{K(K-1)}{2}$ | One for **every pair** of classes |

#### 🧠 Worked example — K = 4 classes

- **OvR:** $K - 1 = 3$ classifiers.
- **OvO:** $\dfrac{4 \times 3}{2} = 6$ classifiers.

#### ✏️ Practice example (made up — solved)

You classify handwritten digits into **K = 10** classes (0–9):
- **One-vs-Rest:** $K - 1 = 10 - 1 = \mathbf{9}$ classifiers (each: "is it a 7 vs not-7?" etc.).
- **One-vs-One:** $\dfrac{K(K-1)}{2} = \dfrac{10 \times 9}{2} = \dfrac{90}{2} = \mathbf{45}$ classifiers (one for every pair like "3 vs 8").

→ OvO needs *far* more classifiers (45 vs 9) as classes grow — a real cost for many-class problems.

> ⚠️ **The ambiguity problem:** both OvR and OvO can produce regions where votes **conflict** — a point claimed by 2 classes or none. The main weakness of naïvely combining binary classifiers.

---

## 5. The K Linear Functions Approach

> 🎯 **Plain idea:** give **each class its own scoring machine**. Show a point to all $K$ machines; each shouts a number; the **loudest one wins**. No vote conflicts, no "?" regions — exactly one winner every time. That's why it beats OvR/OvO.

Build **$K$ linear functions**, one per class:

$$y_k(\mathbf{x}) = \mathbf{w}_k^T\mathbf{x} + w_{k0}$$

**Decision rule:** assign $\mathbf{x}$ to $C_k$ if $y_k(\mathbf{x}) > y_j(\mathbf{x})$ for all $j \neq k$ (the biggest score wins — "winner takes all").

**Boundary between $C_k$ and $C_j$** = where their scores tie:
$$(\mathbf{w}_k - \mathbf{w}_j)^T\mathbf{x} + (w_{k0} - w_{j0}) = 0$$
(Just "set $y_k = y_j$ and move everything to one side." The boundary is where two machines are *tied*.)

> 🔑 **Key property:** the decision regions are always **singly connected and convex** — no fragmented/ambiguous regions like OvR/OvO. (Convex = the line between any two points of a region stays inside it.)

#### 🧠 Worked example — winner-takes-all

3 classes, $y_1 = 2$, $y_2 = 5$, $y_3 = 3$ for a point. Largest is $y_2 = 5$ → **class $C_2$**.

#### ✏️ Practice example (made up — solved)

4 scoring machines for an animal classifier (cat, dog, rabbit, horse) give, for a photo:
$$y_{cat}=1.2,\quad y_{dog}=4.7,\quad y_{rabbit}=0.9,\quad y_{horse}=4.9$$
Largest is $y_{horse} = 4.9$ → predict **horse**. (Even though "dog" was close at 4.7, there's still exactly **one** winner — no tie/ambiguity, the advantage over OvR/OvO.)

---

## 6. Least Squares for Classification

> 🎯 **Plain idea:** we want each machine to output `1` for the correct class and `0` for the others. So write "how far off are we from that ideal?", square it (over- and under-shoots both count), sum over all data, and pick the weights that minimize the total. Same recipe as regression — applied to 0/1 targets.

We use **1-of-K (one-hot) targets**: for 3 classes, class 2 → $\mathbf{t} = (0, 1, 0)$.

Error to minimize over all $N$ examples and $K$ class-slots:

$$E(\mathbf{W}) = \frac{1}{2}\sum_{n=1}^{N}\sum_{k=1}^{K}(y_k(\mathbf{x}_n) - t_{nk})^2$$

| Symbol | Meaning |
|---|---|
| $N$ | number of training examples |
| $K$ | number of classes |
| $y_k(\mathbf{x}_n)$ | model's score for class $k$ on example $n$ |
| $t_{nk}$ | true target (1 if example $n$ is class $k$, else 0) |
| $\frac{1}{2}$ | makes the calculus cleaner when differentiating |

> 💡 Just MSE applied to classification: for every example, for every class slot, square the gap between the predicted score and the 0/1 target, add them all, minimize.

#### 🧠 Worked example — compute the error

One example, 3 classes. True class 2 → $\mathbf{t} = (0, 1, 0)$. Model outputs $\mathbf{y} = (0.2, 0.7, 0.3)$.
$$E = \frac{1}{2}\Big[(0.2-0)^2 + (0.7-1)^2 + (0.3-0)^2\Big] = \frac{1}{2}[0.04 + 0.09 + 0.09] = \frac{1}{2}(0.22) = 0.11$$

#### ✏️ Practice example (made up — solved)

One example, 3 classes. True class **1** → $\mathbf{t} = (1, 0, 0)$. Model outputs $\mathbf{y} = (0.6, 0.3, 0.2)$.
$$E = \frac{1}{2}\Big[(0.6-1)^2 + (0.3-0)^2 + (0.2-0)^2\Big]$$
$$= \frac{1}{2}\big[(-0.4)^2 + 0.09 + 0.04\big] = \frac{1}{2}[0.16 + 0.09 + 0.04] = \frac{1}{2}(0.29) = \mathbf{0.145}$$
→ Sum this over every training example for the total error; the optimizer adjusts $\mathbf{W}$ to make this total smallest.

---

## 7. Problems with Least Squares

### 🎯 Picture this

You're positioning a fence between two herds of sheep. Then one sheep wanders **way, way off** into its own side — already obviously safe. A sensible person ignores it. But least squares **panics** over how far that one sheep is from the "ideal target value" and **rotates the whole fence** toward it — now misclassifying border sheep that were fine before. It punishes being "too correct" as if it were an error.

### 🔗 Why this matters

#### Problem 1 — not robust to far-away "easy" points

- Adding **easy points far from the boundary** makes the boundary **worse**.
- **Why?** If the target is 1 but a far point gives a huge score like 10, then $(10-1)^2 = 81$ is a **massive** squared error. Least squares "fixes" this by rotating the boundary — even though that point was already correct.

```
   Before:  good boundary       After far points added: boundary DRAGGED
   × ×  ╲                       × ×  ╲
   ×× × ╲ o o                   ×× ×  ╲ o o   ← misclassifies now!
          ╲  o (far easy pt)            ╲ o o (pulls the line)
```

> 🔑 **Root cause:** least squares squares *all* deviations from the target, so a point far on the *correct* side looks like a disaster.

#### Problem 2 — masks classes in multi-class

With 3+ classes, least-squares boundaries can **squeeze out the middle class** entirely (the green class gets almost no region — "masked" by red and blue).

> ✅ **Takeaway:** simple but **fragile** — sensitive to outliers, prone to masking. Motivates logistic regression / LDA.

---

## 8. Fisher Linear Discriminant / LDA

> 🎯 **The one idea:** imagine each data point is a star, and you shine all of them onto a flat wall (a "projection"). Depending on the **angle** you shine from, the class-blobs on the wall either **overlap into a mess** or **land in clean separate clumps**. LDA's only job: find the **best angle** so the clumps are far apart *and* each clump stays tight.

> **LDA (Linear Discriminant Analysis)** finds a **projection direction** that **best separates the classes**. Also used for **dimensionality reduction**.

### 8.1 The core idea

Project high-dim points onto a line, then separate on that line. Which direction?

- ❌ **Naïve:** project onto the line joining the two class **means**. If classes are stretched along it, projections **overlap**.
- ✅ **Fisher's criterion:** choose the direction that maximizes:

$$\text{maximize} \quad \frac{\text{between-class separation}}{\text{within-class variance}}$$

```
  BAD separability                 GOOD separability
   ■■●●■   ← overlap when           ■■■   ← clean gap
   ●●●●●     projected              ━━━━
   ──────────►                      ●●●●  ← classes well apart
```

> 💡 **Plain English:** push the **class centres far apart** (big numerator) while keeping **each class tightly clustered** (small denominator). "Spread *between* groups ÷ spread *within* groups."

### 8.2 The setup

$C$ classes, class $i$ has $M_i$ samples, total $M = \sum_{i=1}^{C} M_i$.
- $\mu_i$ = mean of class $i$. $\mu$ = overall mean $= \frac{1}{C}\sum_{i=1}^{C}\mu_i$.

### 8.3 Within-class scatter $S_w$ (want **small** — tight clumps)

$$S_w = \sum_{i=1}^{C}\sum_{j=1}^{M_i}(\mathbf{x}_{ij} - \mu_i)(\mathbf{x}_{ij} - \mu_i)^T$$

**Decode:** `(point − its class mean)` = "how far is this point from its own group's centre." The $(\dots)(\dots)^T$ is just the vector way of squaring (makes it positive). Add up that spread for every point in every class. Everyone near their group centre → small $S_w$.

> 💡 In **1-D** this is literally $\sum(\text{point}-\text{class mean})^2$ — the ordinary "spread" you know. **Small $S_w$ = tight, compact classes.**

### 8.4 Between-class scatter $S_b$ (want **large** — separated centres)

$$S_b = \sum_{i=1}^{C}(\mu_i - \mu)(\mu_i - \mu)^T$$

**Decode:** `(class centre − overall centre)` = "how far is this group's centre from the middle of everything." Square, add over classes. In **1-D**: $\sum(\text{class mean}-\text{overall mean})^2$. **Large $S_b$ = centres well separated.**

> 🔑 **LDA's whole job:** find direction $\mathbf{w}$ that **maximizes $\dfrac{S_b}{S_w}$** → classes far apart *and* tightly packed.

#### 🧠 Worked example — 1-D, 2 classes

Class 1: {1, 2, 3}. Class 2: {6, 7, 8}.

1. Class means: $\mu_1 = \frac{1+2+3}{3} = 2$, $\mu_2 = \frac{6+7+8}{3} = 7$
2. Overall mean: $\mu = \frac{2+7}{2} = 4.5$
3. $S_w$: Class 1 $(1-2)^2+(2-2)^2+(3-2)^2 = 2$; Class 2 $(6-7)^2+(7-7)^2+(8-7)^2 = 2$; $S_w = 2+2 = \mathbf{4}$
4. $S_b = (\mu_1-\mu)^2 + (\mu_2-\mu)^2 = (2-4.5)^2 + (7-4.5)^2 = 6.25 + 6.25 = \mathbf{12.5}$
5. Fisher ratio: $\dfrac{S_b}{S_w} = \dfrac{12.5}{4} = \mathbf{3.125}$

> High ratio (3.125) = well separated relative to internal spread — exactly what LDA maximizes.

#### ✏️ Practice example (made up — solved)

Class A: {2, 4}. Class B: {3, 5}. (Deliberately **overlapping** — watch the ratio collapse.)

1. Means: $\mu_A = \frac{2+4}{2} = 3$, $\mu_B = \frac{3+5}{2} = 4$
2. Overall mean: $\mu = \frac{3+4}{2} = 3.5$
3. $S_w$: Class A $(2-3)^2+(4-3)^2 = 1+1 = 2$; Class B $(3-4)^2+(5-4)^2 = 1+1 = 2$; $S_w = 2+2 = \mathbf{4}$
4. $S_b = (3-3.5)^2 + (4-3.5)^2 = 0.25 + 0.25 = \mathbf{0.5}$
5. Fisher ratio: $\dfrac{S_b}{S_w} = \dfrac{0.5}{4} = \mathbf{0.125}$

→ A **low ratio (0.125)** vs the previous **3.125**: these classes barely separate (centres only 1 apart, but each clump spans width 2 — they interleave). LDA would report this as a poor projection direction. Same arithmetic, opposite story — that's the intuition the formula captures.

---

## 9. The Night-Before Recap

**Classification** = sort mail into bins. Assign $\mathbf{x}$ to one of $K$ classes. **Decision boundary** separates **regions**. In $D$ dims, boundary is a **$(D-1)$-dim hyperplane**.

**Generalized linear model** (the bouncer): $y(\mathbf{x}) = f(\mathbf{w}^T\mathbf{x} + w_0)$, $f$ = activation (e.g. step). Boundary still **linear in x**.

**Two-class discriminant:** $y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0$. $y\geq0\to C_1$; $y<0\to C_2$; $y=0\to$ boundary. Sign = side, size = how far.
- Perpendicular distance: $r = \dfrac{y(\mathbf{x})}{\|\mathbf{w}\|}$. Boundary from origin: $\dfrac{-w_0}{\|\mathbf{w}\|}$. ($\|\mathbf{w}\|=\sqrt{w_1^2+\dots}$). Divide by $\|\mathbf{w}\|$ = convert ruler-marks to cm. (Worked: $w=(3,4)$, $x=(2,2)$ → $C_1$, r=0.8, origin dist 2. Practice: $w=(1,2)$, $x=(4,3)$ → $C_1$, r≈2.68.)

**Multi-class** (judging a 4-person race):

| Method | # classifiers |
|---|---|
| One-vs-Rest | $K-1$ |
| One-vs-One | $K(K-1)/2$ |
| K linear funcs | $K$ — pick max $y_k$; regions **convex & singly connected** |

(K=10 digits → OvR 9, OvO 45.)

**Least squares for classification:** minimize $E(\mathbf{W})=\frac12\sum_n\sum_k (y_k-t_{nk})^2$ (one-hot targets). Worked E=0.11; practice E=0.145.
- ❌ The wandering-sheep problem: far "easy" points create huge squared errors → boundary dragged.
- ❌ Masks the middle class in multi-class.

**LDA (Fisher)** = find the best projection *angle* so class-clumps separate cleanly: maximize $\dfrac{S_b}{S_w} = \dfrac{\text{between-class}}{\text{within-class}}$. Also for dimensionality reduction.
- Within-class $S_w = \sum_i\sum_j (\mathbf{x}_{ij}-\mu_i)(\mathbf{x}_{ij}-\mu_i)^T$ — want **small** (tight).
- Between-class $S_b = \sum_i (\mu_i-\mu)(\mu_i-\mu)^T$ — want **large** (separated).
- Worked: {1,2,3} & {6,7,8} → $S_w=4$, $S_b=12.5$, ratio 3.125 (great). Practice: {2,4} & {3,5} → ratio 0.125 (poor — overlapping).

---

*✅ Study guide for Lecture 3 (Discriminant Functions) complete.*
