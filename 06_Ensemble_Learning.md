# Lecture 6: Ensemble Learning

> **Course:** BITS F464 — Machine Learning
> **Covers:** Condorcet's jury theorem, strong vs weak learners, conditions for good ensembles, producing diverse classifiers, randomizing decision trees, aggregation methods, **Bagging** (+ bootstrap), **Random Forests**, **Boosting & AdaBoost** (full 3-round worked example), Bagging vs Boosting.
>
> **How to read this:** Every formula has a worked numerical example. The **AdaBoost 3-round example** (slides 19–37) is the single biggest exam problem in this lecture — I trace every round, every error rate, every weight update, with all numbers.

---

## Table of Contents

1. [The Basic Idea — Condorcet's Jury Theorem](#1-the-basic-idea--condorcets-jury-theorem)
2. [Strong vs Weak Learners](#2-strong-vs-weak-learners)
3. [Conditions for a Good Ensemble](#3-conditions-for-a-good-ensemble)
4. [How to Produce Diverse Classifiers](#4-how-to-produce-diverse-classifiers)
5. [Aggregation Methods](#5-aggregation-methods)
6. [Bagging (+ Bootstrap)](#6-bagging--bootstrap-resampling)
7. [Random Forests](#7-random-forests)
8. [Boosting](#8-boosting)
9. [⭐ AdaBoost — Full Worked Example](#9--adaboost--full-worked-example)
10. [Bagging vs Boosting](#10-bagging-vs-boosting)
11. [Quick Revision Sheet](#11-quick-revision-sheet)

---

## 1. The Basic Idea — Condorcet's Jury Theorem

> **Condorcet's Jury Theorem (1785):** A group votes between two choices (one correct). Each person votes independently and is correct with probability $p$. Combine by **majority rule**. Let $M$ = probability the majority is correct.
> **If $p > 0.5$, then $M \to 1$ as the number of voters → ∞.**

**Meaning:** *"the crowd is cleverer than the individuals"* — under two weak assumptions:
1. Each individual is correct with **$p > 0.5$** (better than random guessing).
2. They make **independent** decisions.

> 🔑 **Ensemble learning** = strategically **generate and combine multiple models** (classifiers/experts) to solve one problem. The theorem is the mathematical justification: combine many "okay" voters → near-perfect group.

#### 🧠 Intuition example

If 1 doctor is right 60% of the time, that's mediocre. But if 100 *independent* doctors each right 60% vote by majority, the **majority is right ~97%+** of the time. Many weak voters → one strong decision.

---

## 2. Strong vs Weak Learners

| Term | Definition |
|---|---|
| **Strong learner** | A classifier whose error can be made **arbitrarily small** (very accurate) |
| **Weak learner** | A classifier just **better than random guessing** (slightly > 50%) |

> 🔑 **The ensemble philosophy:** Instead of building **one strong classifier** (hard), build a **huge set of weak classifiers** (easy) and combine them. By Condorcet's theorem, under proper conditions the ensemble's error → arbitrarily close to **zero**.

> 💡 Creating one perfect model is very hard. Creating lots of "just okay" models and voting is much easier — and works just as well.

---

## 3. Conditions for a Good Ensemble

⚠️ Training the **same** classifier on the **same** data repeatedly gives the **same result** (for deterministic algorithms) → combining identical models is **pointless**.

**Classifier combination works best when:**
- The classifier outputs for the same input are **diverse**.
- The classifiers operate **independently** (or partially).
- The classifiers have **unique / "local" knowledge** (e.g., trained on different data).
- You need a **formalism to aggregate** their opinions.

> 🔑 **DIVERSITY is the key requirement.** Identical classifiers add nothing. The members must disagree in useful, independent ways (echoes Condorcet's "independent voters").

---

## 4. How to Produce Diverse Classifiers

| Strategy | How |
|---|---|
| **Hybridization** | Combine **different algorithms** (GMM + SVM + k-NN…) on the same data |
| **Random restarts** | Same algorithm trained multiple times — **only works if training has randomness** (e.g., NN with different random initialization) |
| **Different data subsets** | Same algorithm on **different subsets** of training data |
| **Different feature subsets** | Use different feature subsets ("random subspace") |
| **Different class subsets** | Useful for multi-class with many classes |
| **Different instance weighting** | Same algo + same data but **weight instances differently** |

### Randomizing decision trees (slide 8)

Decision trees are **the popular choice** for ensembles: small trees are weak classifiers but **train fast & easily**. But the tree algorithm is **deterministic** — to get diverse trees:

1. **Data randomization** — different training-data subsets per tree.
2. **Random subspace** — each tree grown on a random subset of **features**.
3. **Algorithm randomization** — instead of the *best* split attribute, pick **randomly from the K best**.

> 🔑 These three (especially data + random subspace) are exactly what **Random Forests** use.

---

## 5. Aggregation Methods

How to combine the base classifiers' outputs:

**When output is a class label:**
- **Majority voting** — most-voted class wins.
- **Weighted majority voting** — weight each classifier by its reliability.

**When output is numeric** (e.g., probability $d_{ji}$ for class $i$ from classifier $j$):

| Rule | Fusion function $f(\cdot)$ |
|---|---|
| Sum | $y_i = \frac{1}{L}\sum_{j=1}^{L} d_{ji}$ |
| Weighted sum | $y_i = \sum_j w_j d_{ji}$, $w_j\geq 0$, $\sum w_j = 1$ |
| Median | $y_i = \text{median}_j\, d_{ji}$ |
| Minimum | $y_i = \min_j d_{ji}$ |
| Maximum | $y_i = \max_j d_{ji}$ |
| Product | $y_i = \prod_j d_{ji}$ |

**Stacking:** instead of a simple rule, train **another classifier** on the base classifiers' outputs.

#### 🧠 Worked example — combining 3 classifiers

3 classifiers give P(class=spam) = 0.8, 0.6, 0.7.
- **Sum/mean rule:** $(0.8+0.6+0.7)/3 = \mathbf{0.70}$
- **Max rule:** $\max(0.8,0.6,0.7) = \mathbf{0.80}$
- **Product rule:** $0.8\times0.6\times0.7 = \mathbf{0.336}$
- **Majority vote** (threshold 0.5: all say spam) → **spam**

---

## 6. Bagging (+ Bootstrap Resampling)

> **Bagging = Bootstrap + Aggregating.**

**Steps:**
1. Use **bootstrap resampling** to generate $L$ different training sets from the original.
2. Train $L$ base learners (one per set).
3. **Aggregate** by averaging (uniform weights) or **majority voting**.

- Diversity is **not controlled** — left to chance / instability of the base learner.
- Ensemble is **almost always better than a single learner IF the base learners are unstable** (small data change → big model change).

### Bootstrap resampling

> **Bootstrap = take random samples from the original set WITH REPLACEMENT.**

- **Randomness** → needed so the $L$ sets differ.
- **With replacement** → needed to make sets of size $n$ from original of size $n$ (same point can appear multiple times; some omitted).

| Works well with | Doesn't help |
|---|---|
| **Unstable** learners: neural nets, decision trees | **Stable** learners: k-NN, SVM |

### Bagging algorithm summary (slide 13)

```
MODEL GENERATION:
  Let n = number of training instances
  For each of t iterations:
      Sample n instances from training set (WITH REPLACEMENT)
      Apply learning algorithm to the sample
      Store resulting model

CLASSIFICATION:
  For each of the t models:
      Predict class of instance
  Return the class predicted MOST OFTEN  (majority vote)
```

#### 🧠 Worked example — one bootstrap sample

Original data = {A, B, C, D, E} (n=5). One bootstrap sample (random, with replacement):
→ e.g. **{B, B, D, A, E}** — note B appears twice, **C is missing**. Each of the $t$ rounds draws a *different* such sample → diverse trees.

---

## 7. Random Forests

> **Random Forest = an ensemble of many decision trees; output = the MODE (majority vote) of the individual trees' classes.**

**Inputs:** training set $S_n$, $T$ (#trees), $m$ (#variables per split).

**Algorithm:**
1. Choose **$T$** = number of trees to grow.
2. Choose **$m$** = number of variables to split each node. **$m \ll M$** (M = total features); $m$ held **constant** while growing.
3. Grow $T$ trees; for each tree:
   - **(a)** Construct a **bootstrap sample** of size $n$ from $S_n$ **with replacement**; grow a tree from it.
   - **(b)** At each node, select **$m$ variables at random** and use them to find the best split.
   - **(c)** Grow the tree **fully (no pruning)**.
4. To classify point $X$: **collect votes from every tree → majority vote**.

> 🔑 **Random Forest = Bagging + Random Subspace.** Two sources of randomness: (a) bootstrap rows, (b) random $m$ features per node. This double randomness makes the trees diverse → strong ensemble.

#### 🧠 Worked example — Random Forest prediction

5 trees classify a new email: spam, spam, not-spam, spam, not-spam.
→ Votes: spam=3, not-spam=2 → **majority = spam**. (That's the "mode" the definition refers to.)

---

## 8. Boosting

**Problem with Bagging:** diversity is random — no control over whether new classifiers are *useful*.

> **Boosting idea:** generate a **series** of base learners where each one is **forced to focus on the MISTAKES of the previous learner** — so they **complement** each other.

**How (slide 16):** assign a **weight** to each training sample representing its importance:
- Correctly classified → **smaller weight** (less attention next round).
- Misclassified → **larger weight** (focus here next round).

Weights influence the algorithm in two ways:
- **Boosting by sampling:** weights affect the **resampling** (more general).
- **Boosting by weighting:** weights affect the **learner** (only certain learners).

**Aggregation is also smarter:** combine base learners by **weighted voting** — a **better weak classifier gets a larger weight**. Add learners iteratively, increasing combined accuracy each round.

---

## 9. ⭐ AdaBoost — Full Worked Example

**AdaBoost (Adaptive Boosting)** — the most exam-likely problem. Here's the complete slide 19–37 trace.

### The 6 key steps (the boxes at the top of every slide)

| Step | Formula |
|---|---|
| 1. Initialize weights | $w = \dfrac{1}{N}$ (equal for all N points) |
| 2. Error of a weak classifier | $\epsilon = \sum_{\text{wrong}} w_i$ (sum of weights of misclassified points) |
| 3. Pick classifier | the one with the **lowest error rate** |
| 4. Voting power | $\alpha = \dfrac{1}{2}\log\dfrac{1-\epsilon}{\epsilon}$ |
| 5. Final model | $f(x_i) = \sum_{t=1}^{T}\alpha_t h_t(x_i)$ (weighted sum of classifiers) |
| 6. Update weights | $w_{new} = \begin{cases} \dfrac{w_{old}}{2(1-\epsilon)} & \text{if classified correctly} \\[2mm] \dfrac{w_{old}}{2\epsilon} & \text{if classified wrongly} \end{cases}$ |

> 💡 **The intuition:** weak classifiers that are *more accurate* get *more voting power* ($\alpha$). Points that get *misclassified* get their weights *boosted* so the next round focuses on them. Correct points get weights *shrunk*.

### The dataset (slide 19)

5 points: **A, B, D, E = class "+"** (class 1), **C = class "−"** (class 0).
Positions: A(1,5), B(5,5), C(3,3), D(1,1), E(5,1).

```
 6
 5  + A          B +          A,B,D,E = "+"
 4
 3       ▬ C                   C = "−"
 2
 1  + D          E +
 0  1   2   3   4   5
```

The weak classifiers are simple **vertical-line stumps**: X<2, X<4, X<6, X>2, X>4, X>6 (each predicts "+" on one side, "−" on the other).

---

### 🔵 ROUND 1

**Step 1 — Initialize weights:** $N=5$, so each point $w = 1/5$.

| Point | Wa | Wb | Wc | Wd | We |
|---|---|---|---|---|---|
| Weight | 1/5 | 1/5 | 1/5 | 1/5 | 1/5 |

**Step 2 — Error rate of each candidate classifier** ($\epsilon$ = fraction of points wrongly classified, since all weights equal = count/5):

| Classifier | Points it gets Wrong | Error $\epsilon$ |
|---|---|---|
| X < 2 | B & E | 2/5 |
| X < 4 | B, C & E | 3/5 |
| **X < 6** | **C** | **1/5** ← lowest! |
| X > 2 | A, D & C | 3/5 |
| X > 4 | A, D | 2/5 |
| X > 6 | A, B, D, E | 4/5 |

**Step 3 — Pick lowest error:** **X < 6** with $\epsilon = 1/5$ (only misclassifies C).

**Step 4 — Voting power:**
$$\alpha = \frac{1}{2}\log\frac{1-\epsilon}{\epsilon} = \frac{1}{2}\log\frac{1-\frac15}{\frac15} = \frac{1}{2}\log\frac{4/5}{1/5} = \frac{1}{2}\log 4$$

So the first classifier in the ensemble: $h(x) = \frac{1}{2}\log 4 \cdot F(x<6)$.

**Step 6 — Update weights** ($\epsilon = 1/5$, $w_{old}=1/5$):

- **Correctly classified (A, B, D, E):** $w_{new} = \dfrac{w_{old}}{2(1-\epsilon)} = \dfrac{1/5}{2(1-1/5)} = \dfrac{1/5}{2(4/5)} = \dfrac{1/5}{8/5} = \dfrac{1}{8}$
- **Wrongly classified (C):** $w_{new} = \dfrac{w_{old}}{2\epsilon} = \dfrac{1/5}{2(1/5)} = \dfrac{1/5}{2/5} = \dfrac{1}{2}$

| Point | Wa | Wb | Wc | Wd | We |
|---|---|---|---|---|---|
| New weight | 1/8 | 1/8 | **1/2** | 1/8 | 1/8 |

> 👉 Notice: C (the mistake) jumped from 1/5 → **1/2** (huge focus next round). The correct points dropped to 1/8.

---

### 🔵 ROUND 2

**Step 2 — Recompute error with new weights** ($\epsilon = \sum_{\text{wrong}} w_i$, using the new weights):

| Classifier | Wrong | Error $\epsilon$ (sum of their weights) |
|---|---|---|
| **X < 2** | B & E | 1/8 + 1/8 = **2/8** ← candidate |
| X < 4 | B, C & E | 1/8+1/2+1/8 = 6/8 |
| X < 6 | C | 4/8 |
| X > 2 | A, C & D | 1/8+1/2+1/8 = 6/8 |
| **X > 4** | A, D | 1/8 + 1/8 = **2/8** ← candidate |
| X > 6 | A, B, D, E | 4/8 |

Two tie at **2/8** (X<2 and X>4). Slide **picks the first → X < 2**.

**Step 4 — Voting power** (here $\epsilon = 2/8 = 1/4$):
$$\alpha = \frac{1}{2}\log\frac{1-\frac14}{\frac14} = \frac{1}{2}\log\frac{3/4}{1/4} = \frac{1}{2}\log 3$$

Ensemble so far: $h(x) = \frac{1}{2}\log 4 \cdot F(x<6) + \frac{1}{2}\log 3 \cdot F(x<2)$.

**Step 6 — Update weights** ($\epsilon = 2/8$). X<2 misclassifies **B and E**:
- Correct points (A, C, D): $w_{new} = \dfrac{w_{old}}{2(1-\epsilon)}$
- Wrong points (B, E): $w_{new} = \dfrac{w_{old}}{2\epsilon}$

After plugging in and **normalizing**, the slide's resulting table:

| Point | Wa | Wb | Wc | Wd | We |
|---|---|---|---|---|---|
| New weight | 1/12 | 3/12 | 4/12 | 1/12 | 3/12 |

> 👉 B and E (this round's mistakes) got boosted to 3/12; C stays heavy at 4/12.

---

### 🔵 ROUND 3

**Step 2 — Recompute error with Round-2 weights:**

| Classifier | Wrong | Error $\epsilon$ |
|---|---|---|
| X < 2 | B & E | 3/12+3/12 = 1/2 |
| X < 4 | B, C & E | 10/12 |
| X < 6 | C | 4/12 |
| X > 2 | A, C & D | 1/2 |
| **X > 4** | A, D | 1/12+1/12 = **2/12** ← lowest! |
| X > 6 | A, B, D, E | 8/12 |

**Step 3 — Pick lowest:** **X > 4** with $\epsilon = 2/12 = 1/6$.

**Step 4 — Voting power** ($\epsilon = 1/6$):
$$\alpha = \frac{1}{2}\log\frac{1-\frac16}{\frac16} = \frac{1}{2}\log\frac{5/6}{1/6} = \frac{1}{2}\log 5$$

### Final ensemble (slide 37)

$$\boxed{\,h(x) = \tfrac{1}{2}\log 4 \cdot F(x<6) \;+\; \tfrac{1}{2}\log 3 \cdot F(x<2) \;+\; \tfrac{1}{2}\log 5 \cdot F(x>4)\,}$$

| Classifier | What it misses | Voting power |
|---|---|---|
| X < 6 | misses **C** | $\frac12\log 4$ |
| X < 2 | misses **B and E** | $\frac12\log 3$ |
| X > 4 | misses **A and D** | $\frac12\log 5$ |

> 🔑 **The punchline (slide 37):** Each classifier misses a *different* set of points (C / B&E / A&D). When combined, *"any two added together is greater than the third"* — so for **every** point, the majority of weighted votes is correct → **the ensemble classifies all 5 points correctly**, even though no single weak stump could.

> ✅ **Exam recipe (memorize):** ① init $w=1/N$ → ② $\epsilon=\sum_{\text{wrong}}w$ for each stump → ③ pick min $\epsilon$ → ④ $\alpha=\frac12\log\frac{1-\epsilon}{\epsilon}$ → ⑤ update: correct $\to \frac{w}{2(1-\epsilon)}$, wrong $\to \frac{w}{2\epsilon}$ → ⑥ repeat. Final = $\sum\alpha_t h_t$.

---

## 10. Bagging vs Boosting

**This comparison is a guaranteed exam question.**

| Aspect | **Bagging** | **Boosting** |
|---|---|---|
| # learners | N learners | N learners |
| Data per learner | Random sampling **with replacement** (uniform) | Random sampling with replacement **over WEIGHTED data** |
| How learners built | **Parallel** (independent, all at once) | **Sequential** (each fixes the previous one's mistakes) |
| Aggregation | **Simple average** / majority vote ($e=\frac1N\sum e_i$) | **Weighted average** ($e=\sum w_i e_i$) |
| Goal | Reduce **variance** | Reduce **bias** (and variance) |
| Diversity | Left to chance | Forced (focus on errors) |

```
  SINGLE          BAGGING (parallel)        BOOSTING (sequential)
  [data]            [data]→model1            [data]→model1
    ↓                [data]→model2  ─┐        ↓ (reweight errors)
  model              [data]→model3  ─┼─avg   [data]→model2
    ↓                                 ↓        ↓ (reweight errors)
  result          simple average     [data]→model3 → weighted avg
```

> 🔑 **The one-line difference:** **Bagging = parallel + equal weights + simple average** (reduces variance). **Boosting = sequential + reweight mistakes + weighted average** (reduces bias). Random Forest is a bagging method; AdaBoost is a boosting method.

---

## 11. Quick Revision Sheet

**Condorcet's Jury Theorem:** if each voter $p>0.5$ & independent, majority → 100% as voters → ∞. Justifies ensembles.

**Strong vs weak:** strong = arbitrarily small error; weak = just > random. Ensemble = many weak → ~zero error.

**Good ensemble needs:** DIVERSITY — classifiers must be different & independent.

**Diversity via:** different algorithms, random init, different data/feature/class subsets, instance weighting.

**Aggregation:** class label → (weighted) majority vote; numeric → sum/mean/median/min/max/product; stacking → train a meta-classifier.

**Bagging = Bootstrap + Aggregating.** Bootstrap = sample n WITH replacement. Parallel, simple average/vote. Helps **unstable** learners (trees, NN), not stable (kNN, SVM). Reduces **variance**.

**Random Forest** = many trees, output = **mode** (majority). = Bagging + random-$m$-features-per-node. Grow fully, no pruning.

**Boosting:** sequential; each learner focuses on previous mistakes. Misclassified → bigger weight. Weighted voting; better classifier → bigger $\alpha$. Reduces **bias**.

**AdaBoost steps:** ① $w=1/N$ ② $\epsilon=\sum_{wrong}w$ ③ pick min $\epsilon$ ④ $\alpha=\frac12\log\frac{1-\epsilon}{\epsilon}$ ⑤ update correct$=\frac{w}{2(1-\epsilon)}$, wrong$=\frac{w}{2\epsilon}$ ⑥ repeat; final $f=\sum\alpha_t h_t$.
- Worked: Round1 X<6 ($\epsilon$=1/5, $\alpha=\frac12\log4$, C→1/2); Round2 X<2 ($\alpha=\frac12\log3$); Round3 X>4 ($\alpha=\frac12\log5$). Combined → all 5 points correct.

**Bagging vs Boosting:** Bagging = parallel, equal weights, simple avg, ↓variance. Boosting = sequential, reweighted, weighted avg, ↓bias.

---

*✅ Study guide for Lecture 6 (Ensemble Learning) complete. **🎉 All 6 PPTs done — your full ML study guide set is ready!***
