# Lecture 4: Logistic Regression, Decision Trees & Accuracy Metrics

> **Course:** BITS F464 — Machine Learning
> **This chapter has two parts** (from two PPTs):
> - **Part A:** Perceptron, Logistic Regression, Gradient Descent
> - **Part B:** Decision Trees (ID3, Entropy, Information Gain) + Classification Accuracy Metrics
>
> **How to read this:** Every formula has a worked numerical example. The sigmoid calc, cross-entropy cost, gradient-descent update, the full **ID3 PlayTennis** calculation, and the confusion-matrix metrics are the exam-likely problems — all worked end-to-end.

---

# ═══════════ PART A: Perceptron, Logistic Regression, Gradient Descent ═══════════

---

## Table of Contents

1. [Perceptron](#1-perceptron)
2. [Perceptron Learning Algorithm](#2-perceptron-learning-algorithm)
3. [Logistic Regression — what & when](#3-logistic-regression--what--when)
4. [Linear vs Logistic Regression](#4-linear-vs-logistic-regression)
5. [The Sigmoid Function](#5-the-sigmoid-function)
6. [Training Process & Cost Function](#6-training-process--cost-function)
7. [Gradient Descent](#7-gradient-descent)
8. [Types of Gradient Descent](#8-types-of-gradient-descent)
9. [Quick Revision Sheet](#9-quick-revision-sheet)

---

## 1. Perceptron

The perceptron is the **simplest artificial neuron** — the ancestor of all neural networks.

### History (exam fact)

| Year | Who | What |
|---|---|---|
| 1958 | **Frank Rosenblatt** | Proposed the classical perceptron |
| 1969 | **Minsky & Papert** | Refined it → the "Perceptron model" |

### McCulloch-Pitts Neuron vs Perceptron

**McCulloch-Pitts neuron** (the older, cruder model — *no weights*):
$$y = 1 \;\text{ if } \sum_{i=0}^{n} x_i \geq \theta, \qquad y = 0 \;\text{ if } \sum_{i=0}^{n} x_i < \theta$$
- Just **adds up the inputs** and compares to a threshold $\theta$. Inputs must be **Boolean** (0/1).

**Perceptron** — the key upgrade: **add a weight $w_i$ to each input**:
$$y = 1 \;\text{ if } \sum_{i=0}^{n} w_i x_i \geq \theta, \qquad y = 0 \;\text{ if } \sum_{i=0}^{n} w_i x_i < \theta$$

> 🔑 **THE difference (very common exam question):** the perceptron introduced **numerical weights for inputs + a mechanism to learn those weights**. Inputs are **no longer limited to Boolean** values.

### The bias trick (slide 4)

Instead of carrying $\theta$ separately, we move it to the other side as a **bias weight**:

$$\sum w_i x_i \geq \theta \quad\Longleftrightarrow\quad \sum w_i x_i - \theta \geq 0$$

Define a dummy input $x_0 = 1$ and weight $w_0 = -\theta$. Then everything collapses to one clean condition:

$$y = 1 \;\text{ if } \sum_{i=0}^{n} w_i x_i \geq 0, \qquad y = 0 \;\text{ otherwise}$$

```
   x₀=1 ─w₀=-θ──┐
   x₁ ───w₁─────┤
   x₂ ───w₂─────┤──►(Σ wᵢxᵢ)──►[ ≥0? ]──► y (0 or 1)
   …             │
   xₙ ───wₙ─────┘
```

> 💡 **Bias = a "prior" $w_0$.** It shifts the decision boundary so it doesn't have to pass through the origin. Weights & bias both **depend on data and must be learned**.

#### 🧠 Worked example

Weights $w = (w_0, w_1, w_2) = (-2, 1, 1)$ with $x_0=1$, input $(x_1, x_2) = (3, 1)$:
$$\sum w_i x_i = (-2)(1) + (1)(3) + (1)(1) = -2 + 3 + 1 = 2$$
Since $2 \geq 0$ → output **$y = 1$**.

---

## 2. Perceptron Learning Algorithm

How does the perceptron *learn* the weights? By **adjusting them whenever it makes a mistake**.

**Setup:** $P$ = inputs labelled 1, $N$ = inputs labelled 0. Initialize $\mathbf{w}$ randomly.

```
while not converged:
    pick a random point x from P ∪ N
    if x ∈ P  and  Σ wᵢxᵢ < 0  (should be +, predicted −):
        w = w + x          ← push boundary toward x
    if x ∈ N  and  Σ wᵢxᵢ ≥ 0  (should be −, predicted +):
        w = w − x          ← push boundary away from x
// converges when ALL inputs classified correctly
```

> 💡 **Intuition:** if a positive point is misclassified, *add* it to $\mathbf{w}$ (tilt the boundary toward classifying it positive). If a negative point is misclassified, *subtract* it. Repeat until no mistakes.

**Absolute linear separability (definition):** two sets $P$ and $N$ in $n$-dim space are *absolutely linearly separable* if real numbers $w_0, \dots, w_n$ exist such that every point in $P$ satisfies $\sum w_i x_i > w_0$ and every point in $N$ satisfies $\sum w_i x_i < w_0$.

> ⚠️ The perceptron algorithm **only converges if the data is linearly separable**. (This is the famous limitation Minsky & Papert pointed out — it fails on XOR.)

#### 🧠 Worked example — one update step

Current $\mathbf{w} = (0, 1, -1)$ (incl. bias). A point $\mathbf{x} = (1, 2, 2)$ is in $P$ (true label 1).
- Score $= (0)(1) + (1)(2) + (-1)(2) = 0 + 2 - 2 = 0$. Hmm, $0 \geq 0$ so it predicts 1 → **correct, no update**.

Now suppose $\mathbf{x} = (1, 1, 3) \in P$ with $\mathbf{w} = (0, 1, -1)$:
- Score $= 0 + 1 - 3 = -2 < 0$ → predicted 0 but should be 1 → **misclassified**.
- Update: $\mathbf{w}_{new} = \mathbf{w} + \mathbf{x} = (0+1,\; 1+1,\; -1+3) = \mathbf{(1, 2, 2)}$.
- Re-check: $(1)(1)+(2)(1)+(2)(3) = 1+2+6 = 9 \geq 0$ → now correct ✓

---

## 3. Logistic Regression — what & when

> **Logistic regression** = a classification algorithm for **categorical (specifically binary) data** that also gives you a **probability**.

Despite the name "regression", it does **classification**. The "regression" part is because internally it fits a linear function, then squashes it into a probability.

**Example:** telecom dataset — predict which customers will **churn** (leave next month). Other examples: probability of heart attack (from age, sex, BMI), likelihood of mortgage default.

### When to use logistic regression (exam list)

- Target is **categorical / binary** (0/1, yes/no, positive/negative).
- You need the **probability** of the prediction (not just the class).
- Data is **linearly separable** (decision boundary is a line / plane / hyperplane).
- You want to **understand the impact of each feature** (the weights are interpretable).

### Two-class setup

$$X \in \mathbb{R}^{m \times n} \quad (m \text{ examples}, n \text{ features}), \qquad y \in \{0, 1\}$$
$$\hat{y} = P(y=1 \mid x), \qquad P(y=0 \mid x) = 1 - P(y=1 \mid x)$$

The decision rule (from the diagram): predict class "yes" when $\theta_0 + \theta_1 x_1 + \theta_2 x_2 > 0$.

---

## 4. Linear vs Logistic Regression

**Why not just use linear regression for classification?** Because its output is an unbounded number (could be −5 or +200), not a probability, and a hard step function is brittle.

| | Linear Regression (for classification) | Logistic Regression |
|---|---|---|
| Model | $\theta^T X = \theta_0 + \theta_1 x_1 + \dots$ | $\sigma(\theta^T X)$ |
| Output | A raw number | A **probability** in [0, 1] |
| Decision | Hard **step function**: $\hat y = 1$ if $\theta^T X \geq 0.5$, else 0 | Smooth sigmoid → then threshold |
| Problem | Sharp jump, no probability, sensitive | Smooth, gives probability |

```
   STEP FUNCTION (linear)          SIGMOID (logistic)
 ŷ │      ┌──────                ŷ │          ╭───────
 1 │      │                      1 │       ╭──╯
   │      │   sharp jump           │     ╭─╯  smooth S-curve
 0 │──────┘                      0 │──╮─╯
   └──────┼────► θᵀX               └────┼────► θᵀX
        0.5                           0
```

> 🔑 **The key sentence (exam):** *Sigmoid (σ) of θᵀx gives the **probability** of a point belonging to a class (logistic regression) instead of the raw **value** of y (linear regression).*

---

## 5. The Sigmoid Function

The sigmoid (a.k.a. logistic function) squashes any real number into the range **(0, 1)**:

$$\sigma(\theta^T X) = \frac{1}{1 + e^{-\theta^T X}}$$

```
 σ(θᵀX)
   1 │              ╭──────────  → σ → 1, P(y=1|x) goes UP
       │           ╭╯
  0.5 │·······╭───╯  ← σ(0) = 0.5 exactly
       │     ╭╯
   0 │─────╯              → σ → 0, P(y=1|x) goes DOWN
       └──┼──────────► θᵀX
          0
```

**Behaviour (memorize these 4 lines):**
- $\theta^T X$ **very big** → $\sigma \to 1$ → $P(y=1\mid x)$ goes **up**.
- $\theta^T X$ **very small** (very negative) → $\sigma \to 0$ → $P(y=1\mid x)$ goes **down**.
- Output range is always **[0, 1]** → that's why it can be read as a probability.
- $\sigma(0) = 0.5$ exactly (the decision threshold).

#### 🧠 Worked example — compute the sigmoid

$\theta = (-1, 2)$ and input $X = (2, 5)$ (so $\theta^T X$ means dot product):
$$\theta^T X = (-1)(2) + (2)(5) = -2 + 10 = 8$$
$$\sigma(8) = \frac{1}{1 + e^{-8}} = \frac{1}{1 + 0.000335} \approx \mathbf{0.9997}$$
→ $P(y=1\mid x) \approx 0.9997$ → very confidently class 1.

*(The slide's own example: $\theta=[-1,2]$, $X=[2,5]$ → $\hat y = \sigma(\dots) = 0.7$ — same setup, just shows the sigmoid produces a probability.)*

---

## 6. Training Process & Cost Function

### The training loop (slide 10)

1. **Initialize** $\theta$.
2. **Calculate** $\hat{y} = \sigma(\theta^T X)$ for a customer.
3. **Compare** $\hat{y}$ with the actual $y$, record the **error**.
4. Calculate error for **all** customers.
5. **Change $\theta$** to reduce the cost.
6. **Go back to step 2** (repeat).

**Slide's mini-example:** $\theta = [-1, 2]$, $X = [2, 5]$ → $\hat y = \sigma([-1,2]\cdot[2,5]) = 0.7$. True $y = 1$. **Error $= 1 - 0.7 = 0.3$.**

### The Cost Function (Cross-Entropy / Log Loss)

For a **single example**, the cost is:

$$\text{Cost}(\hat{y}, y) = \begin{cases} -\log(\hat{y}) & \text{if } y = 1 \\ -\log(1 - \hat{y}) & \text{if } y = 0 \end{cases}$$

Combined over **all $m$ examples** into one formula (this red-boxed one is the most important):

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[\, y^i \log(\hat{y}^i) + (1 - y^i)\log(1 - \hat{y}^i) \,\Big]$$

> 💡 **Why this clever form?** The $y^i$ and $(1-y^i)$ act as **switches**:
> - If true $y = 1$: only the $\log(\hat y)$ term survives → cost $= -\log(\hat y)$. Predict $\hat y$ near 1 → cost ≈ 0 (good). Predict near 0 → cost → ∞ (heavily punished).
> - If true $y = 0$: only the $\log(1-\hat y)$ term survives → cost $= -\log(1-\hat y)$. Same logic mirrored.
>
> It **punishes confident wrong answers very harshly** — much better than squared error for classification.

#### 🧠 Worked example — cross-entropy cost

3 examples. True labels $y = (1, 0, 1)$. Predicted $\hat y = (0.9, 0.2, 0.6)$.

Per-example cost:
- Example 1 ($y=1$): $-\log(0.9) = 0.105$
- Example 2 ($y=0$): $-\log(1-0.2) = -\log(0.8) = 0.223$
- Example 3 ($y=1$): $-\log(0.6) = 0.511$

$$J(\theta) = \frac{1}{3}(0.105 + 0.223 + 0.511) = \frac{0.839}{3} \approx \mathbf{0.28}$$
(Using natural log. Lower $J$ = better model.)

---

## 7. Gradient Descent

> **Gradient Descent (GD)** = use the **derivative (slope) of the cost function** to repeatedly nudge the parameters in the direction that **reduces the cost**.

**Picture:** the cost $J(\theta)$ is a **bowl-shaped surface**. You're standing somewhere on the slope. The **gradient** points *uphill* (steepest increase). So to go *down* toward the minimum, step in the **opposite** direction of the gradient.

```
   J(θ)
     \                         /
      \  ● start (high cost)  /
       \  ↓ step down        /
        \  ●                /
         \  ↓              /
          \_●_____________/   ← minimum (lowest cost) = best θ
                θ
```

### The gradient vector

$$\nabla J = \left[\frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, \dots, \frac{\partial J}{\partial \theta_k}\right]$$

For logistic regression, each partial derivative works out to:
$$\frac{\partial J}{\partial \theta_1} = -\frac{1}{m}\sum_{i=1}^{m}(y^i - \hat{y}^i)\,x_1^i$$

### The update rule (THE key formula)

$$\boxed{\;\theta_{new} = \theta_{prev} - \eta \,\nabla J\;}$$

| Symbol | Meaning |
|---|---|
| $\theta_{prev}$ | current parameter value |
| $\eta$ (eta) | **learning rate** — how big a step to take |
| $\nabla J$ | gradient (direction of steepest *increase*) |
| **minus sign** | so we move *downhill* (opposite of the gradient) |

> 💡 **Learning rate $\eta$ trade-off:** too small → learning is painfully slow; too big → you overshoot the minimum and may never settle. (Goldilocks problem.)

### Training with GD (slide 13)

1. Initialize parameters randomly.
2. Feed cost function with training set, calculate error.
3. Calculate the **gradient** of the cost.
4. **Update weights**: $\theta_{new} = \theta_{prev} - \eta\nabla J$.
5. Go to step 2 until cost is small enough.
6. Predict the new customer $X$ via $P(y=1\mid x) = \sigma(\theta^T X)$.

#### 🧠 Worked example — one gradient-descent step

Suppose $\theta_{prev} = 0.8$, gradient $\nabla J = 2$, learning rate $\eta = 0.1$:
$$\theta_{new} = 0.8 - (0.1)(2) = 0.8 - 0.2 = \mathbf{0.6}$$
The gradient was positive (cost increasing in the +θ direction), so we moved θ **down** to 0.6 to reduce cost. Repeat thousands of times → converges to the minimum.

---

## 8. Types of Gradient Descent

| Type | How it updates |
|---|---|
| **Batch GD** | Uses **all** training data for each update (accurate but slow) |
| **Stochastic GD (SGD)** | Uses **one** random example per update (fast, noisy) |
| **Mini-batch GD** | Uses a **small batch** per update (best of both — most common) |
| **Adagrad** | Adaptive learning rate per parameter ⎫ |
| **RMSProp** | Root-mean-square propagation     ⎬ used in **Deep Learning** |
| **Adam** | Adaptive Moment Estimation       ⎭ |

> 🔑 The last three (Adagrad, RMSProp, Adam) are marked **DL** in the slide — they're advanced optimizers for deep learning. **Adam** is the most popular default in practice.

---

## 9. Quick Revision Sheet

**Perceptron:** Rosenblatt (1958), refined by Minsky & Papert (1969).
- McCulloch-Pitts: $y=1$ if $\sum x_i \geq \theta$ (no weights, Boolean inputs).
- Perceptron upgrade: **adds learnable weights** $w_i$; inputs no longer Boolean.
- Bias trick: $x_0=1$, $w_0=-\theta$ → $y=1$ if $\sum w_i x_i \geq 0$.
- Learning: misclassified positive → $w = w + x$; misclassified negative → $w = w - x$. Converges **only if linearly separable**.

**Logistic regression:** binary classification giving a **probability**. Use when target is binary, need probability, data linearly separable, want feature impact.

**Linear vs logistic:** linear → raw number + step function; logistic → **sigmoid → probability**.

**Sigmoid:** $\sigma(\theta^T X) = \dfrac{1}{1+e^{-\theta^T X}}$, range [0,1], $\sigma(0)=0.5$. Big θᵀX → 1; small → 0.

**Cost (cross-entropy / log loss):**
$$J(\theta) = -\frac{1}{m}\sum \big[y\log\hat y + (1-y)\log(1-\hat y)\big]$$
Punishes confident wrong predictions harshly. Lower = better.

**Gradient descent update:** $\theta_{new} = \theta_{prev} - \eta\nabla J$.
- $\nabla J$ = gradient (uphill); minus sign → go downhill; $\eta$ = learning rate (too small=slow, too big=overshoot).

**Types of GD:** Batch (all data), Stochastic (1 example), Mini-batch (small batch); Adagrad/RMSProp/Adam (DL optimizers).

---

*✅ Part A complete. Part B (Decision Trees & Metrics) below.*

---

# ═══════════ PART B: Decision Trees & Classification Accuracy Metrics ═══════════

## Table of Contents (Part B)

10. [What is a Decision Tree?](#10-what-is-a-decision-tree)
11. [Decision Tree Classification Task](#11-decision-tree-classification-task)
12. [Decision Tree Learning Steps](#12-decision-tree-learning-steps)
13. [Entropy](#13-entropy)
14. [Information Gain](#14-information-gain)
15. [⭐ ID3 Algorithm — Full PlayTennis Worked Example](#15--id3-algorithm--full-playtennis-worked-example)
16. [Other Decision Tree Algorithms](#16-other-decision-tree-algorithms)
17. [Classification Accuracy Metrics](#17-classification-accuracy-metrics)
18. [Quick Revision Sheet (Part B)](#18-quick-revision-sheet-part-b)

---

## 10. What is a Decision Tree?

> A **decision tree** is a predictive model built from a **branching series of yes/no (Boolean) tests**. You follow branches from the root down to a leaf, which gives the answer.

It's an **inductive learning** task — use specific facts (training examples) to form **general rules**.

**Slide's commute example:** "How long will my commute take?"

```
                    ┌─────────┐
                    │ Leave At│
                    └────┬────┘
        ┌────────────────┼────────────────┐
     10 AM             8 AM              9 AM
        │                │                 │
   ┌────┴───┐          Long          ┌─────┴────┐
   │ Stall? │                        │ Accident?│
   └────┬───┘                        └─────┬────┘
    No ─┴─ Yes                       No ────┴──── Yes
    │       │                        │            │
  (Short)  Long                   Medium         Long
```

**Why use small Boolean tests?** Each test is **simpler** than one giant all-at-once classifier. You break a hard decision into a chain of easy ones.

### Tree vocabulary (memorize)

| Part | Meaning | PlayTennis example |
|---|---|---|
| **Root node** | The first/top test | Outlook |
| **Internal node** | A test on an attribute | Humidity, Wind |
| **Branch** | A value of that attribute | Sunny, Rain, High |
| **Leaf node** | Final class/output | Yes / No |

**Slide's PlayTennis tree:**
```
                  ┌─────────┐
                  │ Outlook │
                  └────┬────┘
       ┌───────────────┼───────────────┐
     Sunny          Overcast          Rain
       │               │               │
  ┌────┴────┐        (Yes)        ┌────┴────┐
  │Humidity │                     │  Wind   │
  └────┬────┘                     └────┬────┘
  High─┴─Normal                  Strong─┴─Weak
   │      │                        │      │
  (No)  (Yes)                     (No)   (Yes)
```

> 💡 **Reading the tree:** to classify "Sunny + High humidity", start at Outlook → take Sunny branch → reach Humidity → take High branch → leaf says **No** (don't play tennis).

---

## 11. Decision Tree Classification Task

The general supervised-learning workflow (slide 4):

```
  TRAINING SET ──► [Tree Induction Algorithm] ──► LEARN MODEL ──► ┐
  (with known                                                     │
   Class labels)                                                  ▼
                                                              ┌───────┐
  TEST SET ───────────────────────────────────────────────►  │ Model │
  (Class = ?)                          APPLY MODEL ◄──────────└───────┘
                                            │
                                            ▼
                                     predicted classes
```

- **Induction** = learning the tree from the training set.
- **Deduction** = applying the learned tree to predict the test set's unknown classes (the `?` rows).

---

## 12. Decision Tree Learning Steps

The recursive recipe:

1. **Choose an attribute** from the dataset.
2. **Calculate its significance** in splitting the data (how well it separates classes).
3. **Split the data** based on the best attribute's values.
4. **Go to step 1** (repeat on each resulting subset).

> 🔑 **The central question:** *Which attribute is "best" to split on?* → Answer: the one with the **highest Information Gain** (which needs **Entropy** first).

---

## 13. Entropy

> **Entropy = a measure of randomness / uncertainty / impurity** in a set. High entropy = very mixed (hard to predict); low entropy = pure (easy to predict).

For events $\{A_1, \dots, A_n\}$ with probabilities $P(A_i)$:

$$H = -\sum_{i=1}^{n} P(A_i)\log_2 P(A_i)$$
(where $\sum P(A_i) = 1$ and $0 \le P(A_i) \le 1$)

**Self-information** of an event $= -\log_2 P(A_i)$. Entropy is the **weighted average** of the information carried by each event.

**For two classes** (positive & negative examples in set $S$):

$$H(S) = -p_{+}\log_2(p_{+}) - p_{-}\log_2(p_{-})$$

where $p_+$ = fraction of positives, $p_-$ = fraction of negatives.

### Why "less likely = more information" (slide 9)

| Event probability | Information $-\log_2 P$ | Meaning |
|---|---|---|
| $P = 1$ (always happens) | $-\log_2(1) = 0$ | No surprise → **no info** |
| $P = 0.5$ (coin flip) | $-\log_2(0.5) = 1$ bit | Max uncertainty |
| $P = 0.001$ (rare) | $-\log_2(0.001) = 9.97$ | Big surprise → **lots of info** |

> 💡 **Intuition:** "The sun rose today" tells you nothing (P≈1). "It snowed in Dubai today" is hugely informative (P≈0). Rare events carry more information.

#### 🧠 Worked example — entropy of a set

A set with 9 positives, 5 negatives (total 14):
$$p_+ = \frac{9}{14}, \quad p_- = \frac{5}{14}$$
$$H(S) = -\frac{9}{14}\log_2\!\frac{9}{14} - \frac{5}{14}\log_2\!\frac{5}{14}$$
$$= -(0.643)(-0.637) - (0.357)(-1.486) = 0.410 + 0.530 = \mathbf{0.940}$$

> **Sanity checks:** A **pure** set (all positive) → $H = -1\log_2 1 - 0 = \mathbf{0}$ (no uncertainty). A **50-50** split → $H = \mathbf{1}$ (max uncertainty). Our 0.940 is high (set is quite mixed, slightly more positives).

---

## 14. Information Gain

> **Information Gain** = how much an attribute **reduces entropy** (uncertainty) when you split on it. The bigger the gain, the better that attribute classifies the data.

$$\text{Gain}(S, A_i) = \underbrace{H(S)}_{\text{entropy before split}} - \underbrace{\sum_{v \in \text{Values}(A_i)} P(A_i = v)\,H(S_v)}_{\text{weighted entropy after split}}$$

| Part | Meaning |
|---|---|
| $H(S)$ | entropy **before** splitting (the mess you start with) |
| $P(A_i = v)$ | fraction of data with attribute value $v$ |
| $H(S_v)$ | entropy of the subset with value $v$ |
| The sum | the **weighted average leftover entropy** after the split |

> 💡 **Plain English:** Gain = (uncertainty before) − (uncertainty after). Pick the attribute that **removes the most uncertainty** → that becomes the node.

---

## 15. ⭐ ID3 Algorithm — Full PlayTennis Worked Example

**This is the #1 most exam-likely problem in this chapter.** ID3 (Iterative Dichotomiser 3) builds the tree by repeatedly picking the highest-information-gain attribute.

**The PlayTennis dataset (14 days):**

| Day | Outlook | Temp | Humidity | Wind | Play? |
|---|---|---|---|---|---|
| D1 | Sunny | Hot | High | Weak | No |
| D2 | Sunny | Hot | High | Strong | No |
| D3 | Overcast | Hot | High | Weak | Yes |
| D4 | Rain | Mild | High | Weak | Yes |
| D5 | Rain | Cool | Normal | Weak | Yes |
| D6 | Rain | Cool | Normal | Strong | No |
| D7 | Overcast | Cool | Normal | Strong | Yes |
| D8 | Sunny | Mild | High | Weak | No |
| D9 | Sunny | Cool | Normal | Weak | Yes |
| D10 | Rain | Mild | Normal | Weak | Yes |
| D11 | Sunny | Mild | Normal | Strong | Yes |
| D12 | Overcast | Mild | High | Strong | Yes |
| D13 | Overcast | Hot | Normal | Weak | Yes |
| D14 | Rain | Mild | High | Strong | No |

**Totals:** 9 "Yes", 5 "No", out of 14.

### STEP 1 — Entropy of the whole set

$$H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = \mathbf{0.940}$$

### STEP 2 — Gain for attribute **Outlook**

Outlook has 3 values. Split the 14 rows:

| Outlook | #Yes | #No | Total | Entropy of subset |
|---|---|---|---|---|
| **Sunny** | 2 | 3 | 5 | $-\frac{2}{5}\log_2\frac25 - \frac{3}{5}\log_2\frac35 = \mathbf{0.971}$ |
| **Overcast** | 4 | 0 | 4 | $-\frac{4}{4}\log_2 1 - 0 = \mathbf{0.0}$ (pure!) |
| **Rain** | 3 | 2 | 5 | $-\frac{3}{5}\log_2\frac35 - \frac{2}{5}\log_2\frac25 = \mathbf{0.971}$ |

**Information Gain:**
$$\text{Gain}(S,\text{Outlook}) = H(S) - \frac{5}{14}H(\text{Sunny}) - \frac{4}{14}H(\text{Overcast}) - \frac{5}{14}H(\text{Rain})$$
$$= 0.940 - \frac{5}{14}(0.971) - \frac{4}{14}(0) - \frac{5}{14}(0.971)$$
$$= 0.940 - 0.347 - 0 - 0.347 = \mathbf{0.246}$$

### STEP 3 — Gain for the other attributes

(Same process, looping over each attribute's values — slide gives the results:)

| Attribute | Information Gain |
|---|---|
| **Outlook** | **0.246** ← biggest |
| Temperature | 0.029 |
| Humidity | 0.029 |
| Wind | 0.048 |

### STEP 4 — Pick the root

> ✅ **Outlook has the highest gain (0.246) → Outlook becomes the ROOT node.**

### STEP 5 — Recurse on each branch

- **Outlook = Overcast** → entropy is **0** (all 4 are "Yes") → it's a **pure leaf → answer: Yes.** Stop here.
- **Outlook = Sunny** (2 Yes, 3 No, $H = 0.971$) → not pure, must split again. Loop over remaining attributes (Temp, Humidity, Wind):

  **Gain(S_Sunny, Temperature):** split the 5 Sunny rows by temp:
  - Hot: 0 Yes, 2 No → H = 0
  - Mild: 1 Yes, 1 No → H = 1.0
  - Cool: 1 Yes, 0 No → H = 0
  $$\text{Gain} = 0.971 - \tfrac{2}{5}(0) - \tfrac{2}{5}(1.0) - \tfrac{1}{5}(0) = 0.971 - 0.4 = \mathbf{0.571}$$

  Similarly: **Gain(S_Sunny, Humidity) = 0.971** (highest!), Gain(S_Sunny, Wind) = 0.020.

  → **Humidity** classifies the Sunny branch best → it becomes the node under Sunny.

- **Outlook = Rain** → repeat the same process → **Wind** wins for that branch.

### STEP 6 — Stopping rules

A branch stops growing when **either**:
- Every attribute has **already been used** along that path, **OR**
- All training examples at that leaf have the **same class** (pure node).

> ⚠️ **Important rule:** an attribute used **higher** in the tree is **excluded** from consideration further down that path.

### Final tree

```
                  ┌─────────┐
                  │ Outlook │  ← root (gain 0.246)
                  └────┬────┘
       ┌───────────────┼───────────────┐
     Sunny          Overcast          Rain
       │               │               │
  ┌────┴────┐        (Yes)        ┌────┴────┐
  │Humidity │     (pure leaf)     │  Wind   │
  └────┬────┘                     └────┬────┘
  High─┴─Normal                 Strong─┴─Weak
   │      │                        │      │
  (No)  (Yes)                     (No)   (Yes)
```

> 🔑 **Exam recipe (memorize this order):** ① compute $H(S)$ → ② for each attribute, split & compute weighted child entropy → ③ Gain = $H(S)$ − weighted child entropy → ④ pick max gain as node → ⑤ recurse on each branch, dropping used attributes → ⑥ stop at pure leaves.

---

## 16. Other Decision Tree Algorithms

| Algorithm | Note |
|---|---|
| **ID3** | Uses Information Gain (what we did above) |
| **CART** | Classification **And** Regression Tree (can predict numbers too; uses Gini index) |
| **C4.5** | Improved ID3 (uses gain ratio, handles continuous data & missing values) |

---

## 17. Classification Accuracy Metrics

After building any classifier, you measure how good it is by comparing **actual labels $y$** vs **predicted labels $\hat y$** on the test set.

### 17.1 Jaccard Index

Measures overlap between actual and predicted label sets:

$$J(y, \hat{y}) = \frac{|y \cap \hat{y}|}{|y \cup \hat{y}|} = \frac{|y \cap \hat{y}|}{|y| + |\hat{y}| - |y \cap \hat{y}|}$$

- $J = 1$ → perfect (sets identical). $J = 0$ → no overlap.

#### 🧠 Worked example (slide's numbers)

10 actual labels, 10 predicted, 8 match:
$$J = \frac{8}{10 + 10 - 8} = \frac{8}{12} = \mathbf{0.66}$$

### 17.2 The Confusion Matrix (foundation for everything else)

```
                    PREDICTED
                  │  Yes  │  No   │
        ──────────┼───────┼───────┤
   A   Yes (1)    │  TP   │  FN   │
   C              │       │       │
   T   ───────────┼───────┼───────┤
   U   No  (0)    │  FP   │  TN   │
   A              │       │       │
   L
```

| Term | Meaning | Memory hook |
|---|---|---|
| **TP** (True Positive) | Correctly predicted **present** | Right, said yes |
| **TN** (True Negative) | Correctly predicted **absent** | Right, said no |
| **FP** (False Positive) | **Wrongly** said present | "False alarm" |
| **FN** (False Negative) | **Wrongly** said absent | "Missed it" |

> 💡 First word (True/False) = was the prediction **correct**? Second word (Positive/Negative) = **what** did it predict?

**Slide's confusion matrix:** TP=6, FN=9, FP=1, TN=24.

### 17.3 Precision, Recall, F1-score

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$

$$F1\text{-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Metric | Question it answers |
|---|---|
| **Precision** | Of all I predicted positive, how many really were? (avoids false alarms) |
| **Recall** | Of all actual positives, how many did I catch? (avoids misses) |
| **F1** | Harmonic mean — balances both (good when classes imbalanced) |

#### 🧠 Worked example — using TP=6, FN=9, FP=1, TN=24

$$\text{Precision} = \frac{6}{6+1} = \frac{6}{7} = 0.86$$
$$\text{Recall} = \frac{6}{6+9} = \frac{6}{15} = 0.40$$
$$F1 = 2 \times \frac{0.86 \times 0.40}{0.86 + 0.40} = 2 \times \frac{0.344}{1.26} = 2 \times 0.273 = \mathbf{0.55}$$

(Matches the slide's "Churn=1: precision 0.86, recall 0.40, f1 0.55". The avg accuracy across both classes was 0.72.)

> 💡 **Why F1 not just accuracy?** If 95% of emails are "not spam", a model that always says "not spam" gets 95% accuracy but is useless (recall for spam = 0). F1 exposes that.

### 17.4 Log Loss

> Measures classifier performance when the output is a **probability** (0 to 1), not a hard class.

$$\text{LogLoss} = -\frac{1}{n}\sum\Big[\, y \times \log(\hat{y}) + (1 - y)\times\log(1 - \hat{y}) \,\Big]$$

- This is the **same cross-entropy formula** from Part A (§6)! Used here as an evaluation metric.
- **Lower LogLoss = higher accuracy** (opposite direction to Jaccard/F1, where higher = better).
- Punishes confident wrong probabilities. (Slide: a true label 1 predicted as 0.13 → LogLoss 2.04, very high; predicted 0.91 → LogLoss 0.11, low.)

#### 🧠 Worked example

True $y=1$, predicted $\hat y = 0.9$: contribution $= -[1\cdot\log(0.9) + 0\cdot\log(0.1)] = -\log(0.9) = 0.105$.
True $y=0$, predicted $\hat y = 0.2$: contribution $= -[0 + 1\cdot\log(0.8)] = -\log(0.8) = 0.223$.
Average over the dataset → final LogLoss (slide's example total = 0.60).

> 🔑 **Metric direction summary:** Jaccard, Precision, Recall, F1 → **higher is better**. LogLoss → **lower is better**.

---

## 18. Quick Revision Sheet (Part B)

**Decision Tree:** branching Boolean tests; root → internal nodes → leaves. Inductive learning.

**Decision Tree learning:** choose attribute → measure significance → split → recurse.

**Entropy** (impurity): $H(S) = -p_+\log_2 p_+ - p_-\log_2 p_-$.
- Pure set → H=0. 50-50 → H=1. Rare event carries more info ($-\log_2 P$).
- 9 pos / 5 neg of 14 → H = **0.940**.

**Information Gain** = $H(S) - \sum P(A=v)H(S_v)$ (entropy before − weighted entropy after). Pick **max gain**.

**ID3 PlayTennis (memorize):** H(S)=0.940 → Gain(Outlook)=**0.246** (max), Temp=0.029, Humidity=0.029, Wind=0.048 → **Outlook = root**. Overcast branch is pure (Yes). Recurse: Sunny→Humidity, Rain→Wind. Drop used attributes; stop at pure leaves.

**Other algorithms:** CART (classification+regression, Gini), C4.5 (improved ID3).

**Confusion matrix:** TP / TN (correct), FP (false alarm) / FN (miss).

| Metric | Formula | Direction |
|---|---|---|
| Jaccard | $\frac{|y\cap\hat y|}{|y|+|\hat y|-|y\cap\hat y|}$ | higher better |
| Precision | $\frac{TP}{TP+FP}$ | higher better |
| Recall | $\frac{TP}{TP+FN}$ | higher better |
| F1 | $2\frac{P\cdot R}{P+R}$ | higher better |
| LogLoss | $-\frac1n\sum[y\log\hat y+(1-y)\log(1-\hat y)]$ | **lower** better |

Worked: Jaccard 8/(10+10−8)=0.66. TP6/FN9/FP1/TN24 → P=0.86, R=0.40, F1=0.55.

---

*✅ Study guide for Lecture 4 (Logistic Regression + Decision Trees + Accuracy Metrics) — both parts complete in one file. Send the next PPT whenever ready.*
