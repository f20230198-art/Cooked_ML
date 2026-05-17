# Lecture 6: Ensemble Learning

> **Course:** BITS F464 — Machine Learning
> **Covers:** Condorcet's jury theorem, strong vs weak learners, conditions for good ensembles, producing diverse classifiers, randomizing decision trees, aggregation methods, **Bagging** (+ bootstrap), **Random Forests**, **Boosting & AdaBoost** (full 3-round worked example), Bagging vs Boosting.
>
> **How this chapter is written:** every idea opens with an everyday story, connects it to *why*, then the method/formula with symbols explained in the story's language. All original worked examples kept; important formulas also get an **extra made-up example, fully solved**.

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
9. [AdaBoost — Full Worked Example](#9-adaboost--full-worked-example)
10. [Bagging vs Boosting](#10-bagging-vs-boosting)
11. [The Night-Before Recap](#11-the-night-before-recap)

---

## 1. The Basic Idea — Condorcet's Jury Theorem

### 🎯 Picture this

"Who Wants to Be a Millionaire" — **Ask the Audience**. Any single audience member is just an average person who might be wrong. But poll the *whole* audience and take the majority answer, and it's right an astonishing fraction of the time. A crowd of so-so guessers, combined, beats almost any individual.

### 🔗 Why this matters

> **Condorcet's Jury Theorem (1785):** a group votes between two choices (one correct). Each person votes independently and is correct with probability $p$. Combine by **majority rule**.
> **If $p > 0.5$, then the probability the majority is correct → 1 as the number of voters → ∞.**

The crowd beats the individuals — under two assumptions:
1. Each individual is correct with **$p > 0.5$** (better than a coin flip).
2. They decide **independently**.

> 🔑 **Ensemble learning** = strategically **generate and combine multiple models**. This theorem is the math behind it: many "okay" voters → a near-perfect group.

#### 🧠 Intuition example

1 doctor right 60% of the time = mediocre. But 100 *independent* doctors each right 60%, voting by majority → the **majority is right ~97%+**.

#### ✏️ Practice example (made up — solved)

**3 independent friends** guess a true/false question, each correct with $p = 0.7$. You take the majority (best 2 of 3). What's the chance the majority is right?

The majority is right if **at least 2 of 3** are right. Probabilities (each right = 0.7, wrong = 0.3):
- All 3 right: $0.7 \times 0.7 \times 0.7 = 0.343$
- Exactly 2 right (3 ways): $3 \times (0.7 \times 0.7 \times 0.3) = 3 \times 0.147 = 0.441$

$$P(\text{majority right}) = 0.343 + 0.441 = \mathbf{0.784}$$

→ Three 70%-friends, combined, reach **78.4%** — better than any one of them alone (70%). Add more independent friends and it keeps climbing toward 100%. *That* is the entire promise of ensembles.

---

## 2. Strong vs Weak Learners

### 🎯 Picture this

Building one **genius** who's right about everything is incredibly hard. Recruiting lots of **mediocre helpers** who are each just slightly better than a coin flip is easy — and (by the audience theorem) combining enough of them works just as well as the genius.

### 🔗 Why this matters

| Term | Definition |
|---|---|
| **Strong learner** | Error can be made **arbitrarily small** (very accurate) — the genius |
| **Weak learner** | Just **better than random** (slightly > 50%) — a mediocre helper |

> 🔑 **The ensemble philosophy:** instead of one strong classifier (hard), build many weak ones (easy) and combine them. Under the right conditions the ensemble's error → near zero.

---

## 3. Conditions for a Good Ensemble

### 🎯 Picture this

If you ask the audience but **everyone has the exact same wrong information**, polling them is useless — they'll all be wrong together. The crowd only helps when its members **think differently and independently**.

### 🔗 Why this matters

⚠️ Training the **same** classifier on the **same** data gives the **same result** → combining identical models is pointless.

Combination works best when classifiers are:
- **Diverse** (give different outputs for the same input),
- **Independent** (or partially),
- **Locally knowledgeable** (e.g. trained on different data),
- combined by a **formal aggregation** rule.

> 🔑 **DIVERSITY is the key requirement.** Identical classifiers add nothing — members must disagree in useful, independent ways (Condorcet's "independent voters").

---

## 4. How to Produce Diverse Classifiers

### 🎯 Picture this

To get a panel of genuinely different opinions, you might invite people from **different professions**, or give each one **different evidence**, or have them **focus on different aspects** of the case. Same goal: make them disagree usefully.

### 🔗 Why this matters

| Strategy | How |
|---|---|
| **Hybridization** | Combine **different algorithms** (GMM + SVM + k-NN…) on same data |
| **Random restarts** | Same algorithm, multiple times — only works if training has randomness |
| **Different data subsets** | Same algorithm on **different subsets** of data |
| **Different feature subsets** | Different feature subsets ("random subspace") |
| **Different class subsets** | Useful for many-class problems |
| **Different instance weighting** | Same algo/data but **weight instances differently** |

### Randomizing decision trees

Trees are popular for ensembles: small trees are weak but **train fast**. The tree algorithm is **deterministic**, so to get diverse trees:
1. **Data randomization** — different data subsets per tree.
2. **Random subspace** — random subset of **features** per tree.
3. **Algorithm randomization** — pick **randomly from the K best** split attributes.

> 🔑 These (especially 1 & 2) are exactly what **Random Forests** use.

---

## 5. Aggregation Methods

### 🎯 Picture this

You've collected the panel's opinions — now how do you merge them into one verdict? Take the **most common vote**? **Average** their confidence scores? Take the **most cautious** one? Different merge rules for different situations.

### 🔗 Why this matters

**Output is a class label:**
- **Majority voting** — most-voted class wins.
- **Weighted majority voting** — weight each classifier by reliability.

**Output is numeric** ($d_{ji}$ = classifier $j$'s probability for class $i$):

| Rule | Fusion $f(\cdot)$ |
|---|---|
| Sum/mean | $y_i = \frac{1}{L}\sum_{j=1}^{L} d_{ji}$ |
| Weighted sum | $y_i = \sum_j w_j d_{ji}$, $\sum w_j = 1$ |
| Median | $y_i = \text{median}_j\, d_{ji}$ |
| Minimum | $y_i = \min_j d_{ji}$ |
| Maximum | $y_i = \max_j d_{ji}$ |
| Product | $y_i = \prod_j d_{ji}$ |

**Stacking:** train **another classifier** on the base classifiers' outputs.

#### 🧠 Worked example — combining 3 classifiers

3 classifiers give P(spam) = 0.8, 0.6, 0.7.
- **Mean:** $(0.8+0.6+0.7)/3 = \mathbf{0.70}$
- **Max:** $\max(0.8,0.6,0.7) = \mathbf{0.80}$
- **Product:** $0.8\times0.6\times0.7 = \mathbf{0.336}$
- **Majority vote** (threshold 0.5: all say spam) → **spam**

#### ✏️ Practice example (made up — solved)

4 classifiers give P(fraud) = **0.9, 0.4, 0.6, 0.5**.
- **Mean:** $(0.9+0.4+0.6+0.5)/4 = 2.4/4 = \mathbf{0.60}$
- **Max:** $\max(0.9,0.4,0.6,0.5) = \mathbf{0.90}$ (most alarmed classifier)
- **Min:** $\min(\dots) = \mathbf{0.40}$ (most cautious classifier)
- **Product:** $0.9\times0.4\times0.6\times0.5 = \mathbf{0.108}$
- **Majority vote** (threshold 0.5: classifiers 1 & 3 say fraud (≥0.5), classifier 4 is exactly 0.5, classifier 2 says no) → 3 "fraud" vs 1 "no" → **fraud**

→ Same opinions, very different verdicts depending on the merge rule — that's why the choice of aggregation matters.

---

## 6. Bagging (+ Bootstrap Resampling)

### 🎯 Picture this

To get diverse panellists, give each one a **different random handful of the case files** — drawn from the same big pile, **with replacement** (so the same file can appear twice in one handful, and some files get left out). Each panellist forms their own opinion from their own handful; you then **average / majority-vote** their verdicts.

### 🔗 Why this matters

> **Bagging = Bootstrap + Aggregating.**

1. **Bootstrap resample** → $L$ different training sets.
2. Train $L$ base learners (one per set).
3. **Aggregate** by averaging or majority vote.

**Bootstrap = random sample WITH REPLACEMENT.** Randomness → the $L$ sets differ. With replacement → you can make size-$n$ sets from a size-$n$ original (some points repeat, some omitted).

| Works well with | Doesn't help |
|---|---|
| **Unstable** learners: neural nets, decision trees | **Stable** learners: k-NN, SVM |

```
MODEL GENERATION:
  For each of t iterations:
      Sample n instances WITH REPLACEMENT
      Train a model on the sample; store it
CLASSIFICATION:
  Each model predicts; return the MOST OFTEN predicted class
```

#### 🧠 Worked example — one bootstrap sample

Data = {A, B, C, D, E} (n=5). One bootstrap sample → e.g. **{B, B, D, A, E}** (B twice, **C missing**). Each round draws a different such sample.

#### ✏️ Practice example (made up — solved)

Original data = {1, 2, 3, 4, 5, 6} (n=6). Draw **two** bootstrap samples (size 6, with replacement):

- Sample 1: **{2, 2, 4, 1, 6, 4}** → contains 2 twice, 4 twice; **missing 3 and 5**.
- Sample 2: **{1, 3, 3, 5, 6, 1}** → contains 1 twice, 3 twice; **missing 2 and 4**.

→ Train one tree on Sample 1, another on Sample 2. Because the samples differ (different repeats, different omissions), the two trees come out **different** → diversity, exactly what bagging needs. (Statistical fact: on average a bootstrap sample leaves out about **37%** of the original points.)

---

## 7. Random Forests

### 🎯 Picture this

A Random Forest is bagging's panel of trees, but with an **extra twist for even more diversity**: not only does each tree see a different random handful of *rows* (bootstrap), at **every question** in the tree it's only allowed to consider a **random subset of the features**. Double randomness → very different trees → a strong crowd.

### 🔗 Why this matters

> **Random Forest = an ensemble of many decision trees; output = the MODE (majority vote) of the trees.**

**Algorithm:**
1. Choose **$T$** = number of trees.
2. Choose **$m$** = features considered per split, with **$m \ll M$** (M = total features), held constant.
3. For each of $T$ trees:
   - (a) Build a **bootstrap sample** of size $n$ (with replacement); grow a tree.
   - (b) At each node, pick **$m$ random features**, split on the best of those.
   - (c) Grow **fully (no pruning)**.
4. Classify $X$: **majority vote across all trees**.

> 🔑 **Random Forest = Bagging + Random Subspace.** Two randomness sources: (a) bootstrap rows, (b) random $m$ features per node.

#### 🧠 Worked example — Random Forest prediction

5 trees classify an email: spam, spam, not-spam, spam, not-spam → votes spam=3, not-spam=2 → **spam** (the "mode").

#### ✏️ Practice example (made up — solved)

A forest of **7 trees** classifies a transaction as fraud/legit. Their votes: fraud, legit, fraud, fraud, legit, fraud, fraud.
- Tally: **fraud = 5**, legit = 2.
- Majority (mode) → predict **fraud**.

Also: if there are $M = 16$ total features, a common choice is $m \approx \sqrt{M} = \sqrt{16} = \mathbf{4}$ features considered per node — far fewer than 16, which is what forces the trees to be different from each other.

---

## 8. Boosting

### 🎯 Picture this

A student studying for an exam **smartly**: after each practice test they don't re-study everything equally — they **spend the next session on exactly the questions they got wrong**. Each study round is *forced* to focus on the previous round's mistakes. Over rounds, the weak spots get covered.

### 🔗 Why this matters

**Bagging's weakness:** diversity is random — no control over whether new classifiers are *useful*.

> **Boosting idea:** build a **series** of learners where each is **forced to focus on the previous learner's MISTAKES** — so they **complement** each other.

Assign each training sample a **weight** = its importance:
- Correctly classified → **smaller weight** (less attention next round).
- Misclassified → **larger weight** (focus here next round).

Weights influence the algorithm via **sampling** (reweight the resample) or **weighting** (reweight the learner). Aggregation is **weighted voting** — a better weak classifier gets a larger weight.

---

## 9. AdaBoost — Full Worked Example

**AdaBoost (Adaptive Boosting)** — the full slide 19–37 trace.

### The 6 key steps

| Step | Formula |
|---|---|
| 1. Initialize weights | $w = \dfrac{1}{N}$ (equal for all N points) |
| 2. Error of a weak classifier | $\epsilon = \sum_{\text{wrong}} w_i$ (weights of misclassified points) |
| 3. Pick classifier | the one with **lowest error** |
| 4. Voting power | $\alpha = \dfrac{1}{2}\log\dfrac{1-\epsilon}{\epsilon}$ |
| 5. Final model | $f(x_i) = \sum_{t=1}^{T}\alpha_t h_t(x_i)$ |
| 6. Update weights | correct: $\dfrac{w_{old}}{2(1-\epsilon)}$;  wrong: $\dfrac{w_{old}}{2\epsilon}$ |

> 💡 **The intuition:** more accurate weak classifiers get more **voting power** ($\alpha$). Misclassified points get their weights **boosted** so the next round focuses on them; correct points get **shrunk** (the smart-studying student).

### The dataset

5 points: **A, B, D, E = "+"** (class 1), **C = "−"** (class 0). Positions: A(1,5), B(5,5), C(3,3), D(1,1), E(5,1).

```
 5  + A          B +          A,B,D,E = "+"
 3       ▬ C                   C = "−"
 1  + D          E +
    1   2   3   4   5
```

Weak classifiers are simple **vertical-line stumps**: X<2, X<4, X<6, X>2, X>4, X>6.

### 🔵 ROUND 1

**Step 1 — weights:** $N=5$, each $w = 1/5$.

**Step 2 — error of each candidate** (all weights equal → ε = count/5):

| Classifier | Wrong | Error $\epsilon$ |
|---|---|---|
| X < 2 | B & E | 2/5 |
| X < 4 | B, C & E | 3/5 |
| **X < 6** | **C** | **1/5** ← lowest |
| X > 2 | A, D & C | 3/5 |
| X > 4 | A, D | 2/5 |
| X > 6 | A, B, D, E | 4/5 |

**Step 3 — pick:** **X < 6**, $\epsilon = 1/5$.

**Step 4 — voting power:**
$$\alpha = \frac{1}{2}\log\frac{1-\frac15}{\frac15} = \frac{1}{2}\log\frac{4/5}{1/5} = \frac{1}{2}\log 4$$

**Step 6 — update weights** ($\epsilon=1/5$, $w_{old}=1/5$):
- Correct (A,B,D,E): $\dfrac{1/5}{2(1-1/5)} = \dfrac{1/5}{8/5} = \dfrac{1}{8}$
- Wrong (C): $\dfrac{1/5}{2(1/5)} = \dfrac{1}{2}$

| Point | A | B | C | D | E |
|---|---|---|---|---|---|
| New weight | 1/8 | 1/8 | **1/2** | 1/8 | 1/8 |

> 👉 C (the mistake) jumped 1/5 → **1/2** (huge focus next round).

### 🔵 ROUND 2

**Step 2 — error with new weights** ($\epsilon = \sum_{\text{wrong}} w$):

| Classifier | Wrong | Error $\epsilon$ |
|---|---|---|
| **X < 2** | B & E | 1/8 + 1/8 = **2/8** ← tie |
| X < 4 | B, C & E | 6/8 |
| X < 6 | C | 4/8 |
| X > 2 | A, C & D | 6/8 |
| **X > 4** | A, D | 1/8 + 1/8 = **2/8** ← tie |
| X > 6 | A, B, D, E | 4/8 |

Tie at 2/8 → pick the first → **X < 2**.

**Step 4:** $\epsilon = 2/8 = 1/4$ → $\alpha = \frac{1}{2}\log\frac{3/4}{1/4} = \frac{1}{2}\log 3$.

**Step 6 — update** (X<2 misclassifies B, E). After normalizing:

| Point | A | B | C | D | E |
|---|---|---|---|---|---|
| New weight | 1/12 | 3/12 | 4/12 | 1/12 | 3/12 |

### 🔵 ROUND 3

**Step 2 — error with Round-2 weights:**

| Classifier | Wrong | Error $\epsilon$ |
|---|---|---|
| X < 2 | B & E | 3/12+3/12 = 1/2 |
| X < 4 | B, C & E | 10/12 |
| X < 6 | C | 4/12 |
| X > 2 | A, C & D | 1/2 |
| **X > 4** | A, D | 1/12+1/12 = **2/12** ← lowest |
| X > 6 | A, B, D, E | 8/12 |

**Step 3 — pick:** **X > 4**, $\epsilon = 2/12 = 1/6$.

**Step 4:** $\alpha = \frac{1}{2}\log\frac{5/6}{1/6} = \frac{1}{2}\log 5$.

### Final ensemble

$$\boxed{\,h(x) = \tfrac{1}{2}\log 4 \cdot F(x<6) \;+\; \tfrac{1}{2}\log 3 \cdot F(x<2) \;+\; \tfrac{1}{2}\log 5 \cdot F(x>4)\,}$$

| Classifier | Misses | Voting power |
|---|---|---|
| X < 6 | **C** | $\frac12\log 4$ |
| X < 2 | **B, E** | $\frac12\log 3$ |
| X > 4 | **A, D** | $\frac12\log 5$ |

> 🔑 **The punchline:** each classifier misses a *different* set (C / B,E / A,D). Combined, *"any two together outweigh the third"* — so for **every** point the weighted majority is correct → **the ensemble classifies all 5 correctly**, even though no single stump could. (The smart-studying student covered every weak spot.)

#### ✏️ Practice example (made up — solved)

Compute the **voting power $\alpha$** for three weak classifiers with errors $\epsilon = 0.5$, $\epsilon = 0.25$, $\epsilon = 0.1$ (using natural log):

- $\epsilon = 0.5$: $\alpha = \frac{1}{2}\log\frac{1-0.5}{0.5} = \frac{1}{2}\log\frac{0.5}{0.5} = \frac{1}{2}\log 1 = \frac{1}{2}(0) = \mathbf{0}$
  → A coin-flip classifier (50% error) gets **zero** voting power. Useless, correctly ignored.
- $\epsilon = 0.25$: $\alpha = \frac{1}{2}\log\frac{0.75}{0.25} = \frac{1}{2}\log 3 = \frac{1}{2}(1.099) \approx \mathbf{0.55}$
- $\epsilon = 0.1$: $\alpha = \frac{1}{2}\log\frac{0.9}{0.1} = \frac{1}{2}\log 9 = \frac{1}{2}(2.197) \approx \mathbf{1.10}$

→ Lower error → much higher voting power (0 → 0.55 → 1.10). AdaBoost automatically lets the **better weak classifiers shout louder** in the final vote.

> ✅ **Recipe:** ① $w=1/N$ → ② $\epsilon=\sum_{\text{wrong}}w$ → ③ pick min $\epsilon$ → ④ $\alpha=\frac12\log\frac{1-\epsilon}{\epsilon}$ → ⑤ update: correct $\to\frac{w}{2(1-\epsilon)}$, wrong $\to\frac{w}{2\epsilon}$ → ⑥ repeat. Final = $\sum\alpha_t h_t$.

---

## 10. Bagging vs Boosting

### 🎯 Picture this

- **Bagging** = many independent panellists working **in parallel**, each on their own random handful, then you average — like several people independently estimating a jar's marble count and you take the mean. Cuts down wild individual errors (**reduces variance**).
- **Boosting** = panellists working **in sequence**, each one studying the previous one's mistakes — like a relay where each runner fixes the last runner's weak leg. Fixes systematic errors (**reduces bias**).

### 🔗 Why this matters

| Aspect | **Bagging** | **Boosting** |
|---|---|---|
| Data per learner | Random w/ replacement (uniform) | Random w/ replacement over **WEIGHTED** data |
| How built | **Parallel** (independent) | **Sequential** (each fixes the previous) |
| Aggregation | **Simple average** / majority ($e=\frac1N\sum e_i$) | **Weighted average** ($e=\sum w_i e_i$) |
| Goal | Reduce **variance** | Reduce **bias** (and variance) |
| Diversity | Left to chance | Forced (focus on errors) |

```
  BAGGING (parallel)        BOOSTING (sequential)
   [data]→model1            [data]→model1
   [data]→model2  ─┐         ↓ (reweight errors)
   [data]→model3  ─┼─avg     [data]→model2
                    ↓          ↓ (reweight errors)
              simple avg       [data]→model3 → weighted avg
```

> 🔑 **One-line difference:** **Bagging = parallel + equal weights + simple average** (↓ variance). **Boosting = sequential + reweight mistakes + weighted average** (↓ bias). Random Forest is bagging; AdaBoost is boosting.

---

## 11. The Night-Before Recap

**Condorcet's Jury Theorem** (Ask the Audience): if each voter $p>0.5$ & independent, majority → 100% as voters → ∞. Justifies ensembles. (Practice: three 70%-friends → 78.4% majority.)

**Strong vs weak:** strong = arbitrarily small error (a genius); weak = just > random (a mediocre helper). Many weak → ~zero error.

**Good ensemble needs DIVERSITY** — classifiers must differ & be independent (a same-info audience is useless).

**Diversity via:** different algorithms, random init, different data/feature/class subsets, instance weighting.

**Aggregation:** class label → (weighted) majority vote; numeric → sum/mean/median/min/max/product; stacking → train a meta-classifier. (Practice: 4 classifiers 0.9/0.4/0.6/0.5 → mean 0.60, max 0.90, product 0.108, majority "fraud".)

**Bagging = Bootstrap + Aggregating** (different random handful per panellist). Bootstrap = sample n WITH replacement (~37% left out). Parallel, simple average/vote. Helps **unstable** learners (trees, NN), not stable (kNN, SVM). Reduces **variance**.

**Random Forest** = many trees, output = **mode** = Bagging + random-$m$-features-per-node. Grow fully, no pruning. ($M=16 \Rightarrow m\approx\sqrt{16}=4$.)

**Boosting** = the smart-studying student: sequential; each learner focuses on previous mistakes. Misclassified → bigger weight. Weighted voting; better classifier → bigger $\alpha$. Reduces **bias**.

**AdaBoost steps:** ① $w=1/N$ ② $\epsilon=\sum_{wrong}w$ ③ min $\epsilon$ ④ $\alpha=\frac12\log\frac{1-\epsilon}{\epsilon}$ ⑤ correct$=\frac{w}{2(1-\epsilon)}$, wrong$=\frac{w}{2\epsilon}$ ⑥ repeat; final $f=\sum\alpha_t h_t$.
- Worked: R1 X<6 ($\epsilon$=1/5, $\alpha=\frac12\log4$, C→1/2); R2 X<2 ($\alpha=\frac12\log3$); R3 X>4 ($\alpha=\frac12\log5$). Combined → all 5 correct.
- Practice (α): $\epsilon$=0.5→α=0 (useless), 0.25→0.55, 0.1→1.10 (better = louder).

**Bagging vs Boosting:** Bagging = parallel, equal weights, simple avg, ↓variance (independent jar-counters). Boosting = sequential, reweighted, weighted avg, ↓bias (relay fixing weak legs).

---

*✅ Study guide for Lecture 6 (Ensemble Learning) complete.*
