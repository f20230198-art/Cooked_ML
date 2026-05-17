# Lecture 2: Linear Regression & Basics

> **Course:** BITS F464 — Machine Learning
> **Covers:** Regression intro, Simple & Multiple Linear Regression (with the full worked θ₀/θ₁ calculation), Model Evaluation (train/test split, k-fold CV), Overfitting/Underfitting, Linear Basis Function models, Bias–Variance trade-off, and Regularization (Ridge L2 / Lasso L1).
>
> **How this chapter is written:** every idea opens with an everyday story, then connects it to *why* we do the thing, then the method/formula with symbols explained in the story's language. All original worked examples kept; important formulas also get an **extra made-up example, fully solved**.

---

## Table of Contents

1. [What is Regression?](#1-what-is-regression)
2. [Types of Regression Models](#2-types-of-regression-models)
3. [Simple Linear Regression](#3-simple-linear-regression)
4. [Worked Example: Finding θ₀ and θ₁](#4-worked-example-finding-θ₀-and-θ₁)
5. [Model Evaluation](#5-model-evaluation)
6. [Multiple Linear Regression](#6-multiple-linear-regression)
7. [Overfitting and Underfitting](#7-overfitting-and-underfitting)
8. [Linear Basis Function Models](#8-linear-basis-function-models)
9. [Bias and Variance](#9-bias-and-variance)
10. [Bias–Variance Trade-off & Learning Curves](#10-biasvariance-trade-off--learning-curves)
11. [Regularization (Ridge & Lasso)](#11-regularization-ridge--lasso)
12. [The Night-Before Recap](#12-the-night-before-recap)

---

## 1. What is Regression?

### 🎯 Picture this

You notice that the **taller** your friends are, the **heavier** they tend to be — not exactly, but there's a trend. If a new friend tells you their height, you could make a sensible **guess at their weight** by following that trend. You're predicting a *number* from another number.

That's **regression**: using a known input to predict a continuous numerical output.

### 🔗 Why this matters

> **Regression = predicting a continuous (numerical) value** (price, temperature, CO₂ emission — *not* a category).

Two vocabulary words you must know:

| Term | Other names | Meaning | In the car example |
|---|---|---|---|
| **Dependent variable (Y)** | target, response, outcome | The thing we **want to predict** | `CO2EMISSIONS` |
| **Independent variable (X)** | predictor, feature | The **input** we predict from | `ENGINESIZE`, `CYLINDERS` |

> 💡 **Memory hook:** Y **dep**ends on X. X is **indep**endent (it's the input that stands alone). In the height→weight story, height = X, weight = Y.

A **regression model** relates Y to a function of X: $Y = f(X)$.

**Example:** Given historical car data, build a model so for a **new car** you can predict its CO₂. Row 9 has `?` for CO₂ — that's exactly what the model fills in.

---

## 2. Types of Regression Models

### 🎯 Picture this

Some trends are **straight** (every extra hour studied adds roughly the same marks). Some **curve** (the first hour of studying helps a lot; the tenth barely moves the needle). And a trend can depend on **one** thing (just study time) or **many** (study time + sleep + attendance).

```
                    REGRESSION MODELS
            ┌──────────────────┴──────────────────┐
      LINEAR Regression                  NON-LINEAR Regression
      (fits a straight line)             (fits a curve)
      ┌────────┴────────┐                ┌────────┴────────┐
   Simple           Multiple          Simple           Multiple
  (1 input)        (many inputs)     (1 input)        (many inputs)
```

| Type | Shape it fits | Example |
|---|---|---|
| **Linear** | Straight line / flat plane | `%Fat = −23.19 + 3.286·BMI − …` |
| **Non-linear** | Curve (S-shape, etc.) | `Mobility = 1288 + 1491·DensityLn + …` |
| **Simple** | Exactly **1** input | Engine size → CO₂ |
| **Multiple** | **Many** inputs | Engine size + cylinders + fuel → CO₂ |

---

## 3. Simple Linear Regression

### 🎯 Picture this

You scatter all your (height, weight) friends as dots on graph paper. They roughly form a slanted band. You take a ruler and draw the **single straight line that best threads through the middle of that cloud**. Now for any new height, you read the line to get the predicted weight.

### 🔗 Why this matters

> **Simple linear regression** = one output, one input, fitted with a **straight line through the cloud of points**.

```
 Emission
   │              •  •   ╱  ← fitted line
   │           •  • ╱ •
   │        • •╱ •  •
   │     •╱ • •           "best straight line
   │  •╱  •                through the data"
   └────────────────────► Engine size
```

### 📐 The line equation (same line, different letters)

All of these are the **exact same equation** — books just use different letters:

| Notation | Slope | Intercept |
|---|---|---|
| $y = mx + c$ | $m$ | $c$ |
| $y = ax + b$ | $a$ | $b$ |
| **$\hat{y} = \theta_0 + \theta_1 x_1$** (this course) | $\theta_1$ | $\theta_0$ |

```
   ŷ      =   θ₀     +    θ₁    ·   x₁
 predicted  intercept    slope    input
```

- **$\theta_0$ = intercept** — where the line crosses the y-axis when $x=0$ (the "starting value").
- **$\theta_1$ = slope** — how much $y$ changes when $x$ goes up by 1 (the "tilt of the ruler").
- Once we **learn** $\theta_0, \theta_1$ from data, we can **predict Y for any new X**.

---

## 4. Worked Example: Finding θ₀ and θ₁

### 🎯 Picture this

Out of the *infinitely many* lines you could draw through the cloud, which ruler-position is "best"? The best line is the one that (a) tilts to match **how strongly y rises with x**, and (b) is shifted vertically to **pass through the cloud's centre of gravity**. The two formulas below compute exactly those two things.

### 📐 The two formulas

**Slope:**
$$\theta_1 = \frac{\sum_{i=1}^{s}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{s}(x_i - \bar{x})^2}$$

**Intercept:**
$$\theta_0 = \bar{y} - \theta_1 \bar{x}$$

| Symbol | Meaning |
|---|---|
| $x_i$ | $i$-th input (here: engine size) |
| $y_i$ | $i$-th output (here: CO₂) |
| $\bar{x}, \bar{y}$ | **mean** (average) of all x's / all y's |
| $s$ | number of data points |

> 💡 **What the slope formula *means*:** the top measures *"do x and y rise together?"* (covariance). The bottom measures *"how spread out is x?"* (variance of x). Slope = how strongly y moves per unit of x. The intercept formula then just slides the line vertically so it passes through the cloud's centre $(\bar x, \bar y)$.

### 🧠 Worked example (the slide's data)

$x$ (engine size) = (2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7)
$y$ (CO₂) = (196, 221, 136, 255, 244, 230, 232, 255, 267)

**STEP 1 — means:**
$$\bar{x} = \frac{2.0 + 2.4 + 1.5 + \dots}{9} = 3.03, \qquad \bar{y} = \frac{196 + 221 + \dots}{9} = 226.22$$

**STEP 2 — slope.** First term, to show the mechanics:
- $(2.0 - 3.03) = -1.03$, $(196 - 226.22) = -30.22$
- product $= (-1.03)(-30.22) = +31.13$ → numerator
- $(2.0 - 3.03)^2 = 1.06$ → denominator

Repeat for all 9 rows, sum top, sum bottom, divide → $\theta_1 = 39$.

**STEP 3 — intercept:**
$$\theta_0 = 226.22 - 39 \times 3.03 = 226.22 - 118.17 = 125.74$$

**STEP 4 — final model:** $\hat{y} = 125.74 + 39 x_1$

**STEP 5 — predict** (row 9, engine size $x = 2.4$):
$$\hat{y} = 125.74 + 39 \times 2.4 = 125.74 + 93.6 = 219.34$$

> ✅ **The procedure to remember:** ① means → ② slope $\theta_1$ → ③ intercept $\theta_0 = \bar y - \theta_1\bar x$ → ④ write $\hat y = \theta_0+\theta_1 x$ → ⑤ substitute new $x$.

### ✏️ Practice example (made up — solved)

Predict a student's exam mark from hours studied. **4 students:**

| $x$ (hours) | $y$ (marks) |
|---|---|
| 1 | 30 |
| 2 | 50 |
| 3 | 60 |
| 4 | 80 |

**Step 1 — means:** $\bar x = \frac{1+2+3+4}{4} = 2.5$, $\bar y = \frac{30+50+60+80}{4} = 55$.

**Step 2 — slope.** Build the table:

| $x_i-\bar x$ | $y_i-\bar y$ | product | $(x_i-\bar x)^2$ |
|---|---|---|---|
| 1−2.5 = −1.5 | 30−55 = −25 | 37.5 | 2.25 |
| 2−2.5 = −0.5 | 50−55 = −5 | 2.5 | 0.25 |
| 3−2.5 = 0.5 | 60−55 = 5 | 2.5 | 0.25 |
| 4−2.5 = 1.5 | 80−55 = 25 | 37.5 | 2.25 |

Numerator $= 37.5 + 2.5 + 2.5 + 37.5 = 80$. Denominator $= 2.25 + 0.25 + 0.25 + 2.25 = 5$.
$$\theta_1 = \frac{80}{5} = \mathbf{16}$$

**Step 3 — intercept:** $\theta_0 = \bar y - \theta_1\bar x = 55 - 16(2.5) = 55 - 40 = \mathbf{15}$.

**Step 4 — model:** $\hat y = 15 + 16x$.

**Step 5 — predict for 5 hours:** $\hat y = 15 + 16(5) = 15 + 80 = \mathbf{95}$ marks.

---

## 5. Model Evaluation

### 🎯 Picture this

A teacher wants to know if a student *truly* learned.
- **Bad:** test them on the *exact* questions they practised — they could've just memorized. Looks like 100%, proves nothing.
- **Better:** hold back some fresh questions they've never seen and test on those.
- **Best:** rotate — split questions into groups, each time hold out a different group as the test, average the scores. No single lucky/unlucky split decides the grade.

### 🔗 Why this matters — the 3 approaches

#### 5.1 Train and Test on the SAME data ❌

Train on all data, test on that same data. **High training accuracy, low real-world accuracy** — it *memorized*. This is the classic cause of **overfitting**.

| Term | Definition |
|---|---|
| **Training accuracy** | % correct on data the model **was trained on** |
| **Out-of-sample accuracy** | % correct on data it **has NOT seen** (the honest number) |

> 💡 Like practising with the exact exam paper — looks perfect, means nothing.

#### 5.2 Train/Test Split ✅

Split into two **mutually exclusive** groups: train on one, test on the other (unseen).

```
   ENTIRE DATASET   ──split──►   TRAINING SET        TESTING SET
   ● ● ● ● ● ● ●                 ● ● ● ● ●            ● ●
                                  (train)             (test, unseen)
```

- ✅ Truly out-of-sample → realistic accuracy.
- ⚠️ Result depends on *which* points landed in train vs test (luck of the split).

#### 5.3 K-Fold Cross-Validation ✅✅ (best)

Split into **k folds**. Each round: one fold = test, the rest = train. Rotate so **every fold is the test set once**. Average all accuracies.

```
 Round 1:  [TEST ] [train] [train] [train]  → 80%
 Round 2:  [train] [TEST ] [train] [train]  → 84%
 Round 3:  [train] [train] [TEST ] [train]  → 82%
 Round 4:  [train] [train] [train] [TEST ]  → 86%
 Final = avg(80,84,82,86) = 83%
```

#### 🧠 Worked example

4 folds give **80%, 84%, 82%, 86%**:
$$\text{Accuracy} = \frac{80 + 84 + 82 + 86}{4} = \frac{332}{4} = 83\%$$

#### ✏️ Practice example (made up — solved)

**5-fold** cross-validation gives accuracies **70%, 75%, 72%, 78%, 80%**:
$$\text{Accuracy} = \frac{70 + 75 + 72 + 78 + 80}{5} = \frac{375}{5} = \mathbf{75\%}$$
→ Reporting **75%** (the average over all 5 rotations) is far more trustworthy than reporting whatever one single split happened to give (which might've been a lucky 80% or an unlucky 70%).

> ✅ **Why best:** every data point is used for *both* training and testing (in different rounds) → consistent, reliable accuracy not dependent on one lucky split.

---

## 6. Multiple Linear Regression

### 🎯 Picture this

Predicting weight from height alone is okay. But weight also depends on **age, diet, exercise…** If you let *several* inputs each contribute their own weighted push, the prediction gets much better. Instead of a line on 2D paper, picture a flat tilted sheet (a plane) in higher dimensions.

### 🔗 Why this matters

Use it when there's **more than one input**. It answers questions like *"Do revision time, anxiety and attendance affect exam performance?"* and *"How much does blood pressure change per unit BMI?"*

### 📐 The equation

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

Compactly, using vectors: $\hat{y} = \theta^T X$, where $\theta^T = [\theta_0, \theta_1, \dots]$ (a row of weights) and $X = [1, x_1, x_2, \dots]^T$ (a column of inputs with a `1` on top).

> 💡 **Why the `1` on top of X?** It pairs with $\theta_0$ so the intercept slots into the same multiplication: $\theta^T X = \theta_0\cdot 1 + \theta_1 x_1 + \dots$ — exactly the long form. The `T` ("transpose") just turns the column of θ's into a row so it can multiply the column X.

#### 🧠 Worked example

Trained model $\hat y = 50 + 30\,x_1 + 15\,x_2$ (CO₂ from engine size $x_1$, cylinders $x_2$). Car with engine size $2.0$, $4$ cylinders:
$$\hat y = 50 + 30(2.0) + 15(4) = 50 + 60 + 60 = 170$$

#### ✏️ Practice example (made up — solved)

House price model: $\hat y = 20 + 5\,x_1 + 3\,x_2 - 2\,x_3$, where $x_1$ = size (100s of sq ft), $x_2$ = bedrooms, $x_3$ = age (years). A house: 12 (×100 sq ft), 3 bedrooms, 10 years old:
$$\hat y = 20 + 5(12) + 3(3) - 2(10) = 20 + 60 + 9 - 20 = \mathbf{69} \text{ (lakhs)}$$
→ Each feature contributes its own weighted push; note age has a *negative* weight (older → cheaper), so it subtracts.

---

## 7. Overfitting and Underfitting

### 🎯 Picture this

Three students prepare for an exam:
- **Underfitter:** barely studied, only knows "the answer is probably C." Does badly on practice *and* the real exam (too simple a strategy).
- **Just right:** understood the concepts. Does well on both.
- **Overfitter:** memorized every practice question's answer **including the typos**. Perfect on practice, but bombs the real exam because the questions changed slightly (memorized noise instead of learning).

### 🔗 Why this matters

> **The central challenge of ML:** perform well on **new, unseen** inputs — called **generalization**.

- **Generalization error** = **test error**. We want it **low**.
- Test error is usually **≥** training error.
- Two goals (and their failures):
  1. Make **training error small** → fail = **underfitting**
  2. Make the **train-test gap small** → fail = **overfitting**

| | **Underfitting** | **Overfitting** |
|---|---|---|
| Definition | Learns **too little** | Fits data **too well** (memorizes noise) |
| Train error | High | Very low |
| Test error | High | High |
| Model type | Too **simple** | Too **complex** |

```
  UNDERFIT (too simple)     GOOD FIT (just right)     OVERFIT (too complex)
  straight line             smooth curve              wiggles through EVERY point
   θ₀+θ₁x                    θ₀+θ₁x+θ₂x²          θ₀+θ₁x+θ₂x²+θ₃x³+θ₄x⁴
```

The overfit curve hits every training point but **wiggles crazily** → predicts terribly on new points (the student who memorized typos).

### 📐 How to fix each

**Reduce OVERFITTING:** cross-validation, more data, fewer features, early stopping, **regularization** (§11).
**Reduce UNDERFITTING:** increase model complexity, add features, remove noise, train longer.

> 🔑 The fixes are **opposites**. Overfit → simplify/penalize. Underfit → add complexity/features.

---

## 8. Linear Basis Function Models

> 🎯 **The whole section in one idea:** a straight line is too weak for data that **bends**. The trick: don't upgrade your ruler — **bend the paper instead**, draw a straight line on the bent paper, then unbend it → you get a curve. "Bending the paper" = pushing the input `x` through some curvy functions *before* the model sees it.

### 8.1 Why a plain straight line isn't enough

$$y = w_0 + w_1 x_1 + \dots + w_D x_D$$

($w$ = weights, same as θ. $D$ = number of inputs.) This can only ever draw a **straight line / flat plane**. If the real data curves, it fits badly.

### 8.2 The trick

🎯 **Story (the ruler & bent paper):** you only own a straight ruler but need to draw a curve. Bend the paper, rule a straight line on it, unbend → curve. Mathematically: feed `x` through a curvy function $\phi(x)$ ("phi") *first*, then fit the straight-line model on *those*:

$$y = w_0 + w_1 \phi_1(x) + w_2 \phi_2(x) + \dots$$

| Symbol | Plain meaning |
|---|---|
| $\phi_j(x)$ | a **basis function** — a fixed curvy recipe on the input (e.g. "square it") |
| $w_j$ | weight (how much of that curvy piece to mix in) — *this* is learned |
| $w_0$ | the offset (θ₀-equivalent) |

> 💡 **The "still linear" point everyone trips on:** we call it *linear* because it's **linear in the weights $w$** (just multiply each $\phi$ by a number and add — easy to solve). It is **not** linear in `x` (φ bent `x` into a curve). "Linear model" means linear in the *weights*, not the input.

With a dummy $\phi_0(x)=1$ it all collapses into $y = \mathbf{w}^T \phi(x)$ — just shorthand for the sum.

### 8.3 Three types of basis functions

> The one thing to remember: **GLOBAL vs LOCAL**:
> - **GLOBAL** = touching the curve in one spot wobbles it *everywhere* (pull one corner of a stiff bedsheet — the whole sheet moves).
> - **LOCAL** = touching one spot only affects *that nearby region* (press one piano key — only that note sounds).

#### (a) Polynomial basis — $\phi_j(x) = x^j$ — GLOBAL

Just raise `x` to powers: $1, x, x^2, x^3,\dots$ = ordinary polynomial regression. Because every point uses the same $x^2,x^3\dots$, nudging the fit in one spot changes the whole curve. **Fix:** split the x-axis into regions, use a small polynomial in each → **splines**.

#### (b) Gaussian basis — LOCAL

$$\phi_j(x) = \exp\left( -\frac{(x - \mu_j)^2}{2s^2} \right)$$

| Symbol | Controls |
|---|---|
| $\mu_j$ ("mu") | **where** the bump sits (its centre) |
| $s$ | **how wide** the bump is |

A **bell-curve bump**: tall near $\mu_j$, ≈0 far away → **LOCAL** (each bump only cares about inputs near its centre). Build a curve by adding bumps at different centres — like covering a shape with overlapping spotlights.

```
  φ(x)  ╱╲   ╱╲   ╱╲   ╱╲    ← each bump local; moving x
        ╱  ╲ ╱  ╲ ╱  ╲ ╱  ╲    only wakes the nearest bumps
       ─────────────────────► x
        μ₁   μ₂   μ₃   μ₄
```

#### (c) Sigmoidal basis — LOCAL

$$\phi_j(x) = \sigma\left( \frac{x - \mu_j}{s} \right), \qquad \sigma(a) = \frac{1}{1 + e^{-a}}$$

A smooth **S-shaped step** from 0 to 1. $\mu_j$ = where the step happens, $s$ = how steep. Treated as **LOCAL** (the "action" is concentrated near $\mu_j$; far away it's flat 0 or flat 1).

#### 🧠 Worked example — compute a sigmoid value

$\mu_j = 0$, $s = 1$, input $x = 2$:
$$a = \frac{x - \mu_j}{s} = \frac{2 - 0}{1} = 2, \qquad \sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1.135} \approx 0.88$$

> Sanity: at $x=\mu$ ($a=0$), $\sigma(0)=0.5$ (midpoint). $x\to\infty\Rightarrow\sigma\to1$; $x\to-\infty\Rightarrow\sigma\to0$. The S-shape.

#### ✏️ Practice example (made up — solved)

A **Gaussian** basis function with centre $\mu_j = 5$ and width $s = 2$. Evaluate it at two inputs:

*At $x = 5$ (right at the centre):*
$$\phi = \exp\left(-\frac{(5-5)^2}{2(2)^2}\right) = \exp(0) = \mathbf{1}$$ (maximum — the peak of the bump).

*At $x = 9$ (far from centre):*
$$\phi = \exp\left(-\frac{(9-5)^2}{2(4)}\right) = \exp\left(-\frac{16}{8}\right) = \exp(-2) \approx \mathbf{0.135}$$ (small — the bump has nearly faded out this far away).

→ This is exactly why Gaussian is **LOCAL**: it's ≈1 at its own centre and quickly dies off, so it only influences inputs *near* $\mu_j = 5$.

> 🔑 **Polynomial = global** (one change affects everywhere). **Gaussian & Sigmoidal = local** (one change affects only nearby).

---

## 9. Bias and Variance

### 🎯 Picture this

> **An archer shooting at a target.** Retraining the model = the archer shoots a fresh batch of arrows.
> - **Bias** = how far the *average* arrow lands from the bullseye. Always landing low-left = high bias (aimed wrong → too simple → **underfit**).
> - **Variance** = how *scattered* the arrows are from each other. Sprayed all over = high variance (jumpy, over-reacts to which data it saw → too complex → **overfit**).
>
> The formulas are literally just "average miss" (bias) and "spread" (variance) in math.

### 9.1 Bias

$$\text{Bias}(\hat{f}(x)) = E[\hat{f}(x)] - f(x)$$

($f$ = true model, $\hat f$ = our estimate, $E[\cdot]$ = "the average of".) **High bias** → oversimplifies → **UNDERFITTING**. Simple model → high bias; complex → low bias.

### 9.2 Variance

$$\text{Variance}(\hat{f}(x)) = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$$

**High variance** → over-reacts to training data (incl. noise) → great on train, bad on test → **OVERFITTING**. Simple → low variance; complex → high variance.

| | Bias | Variance |
|---|---|---|
| **Simple model** | **High** | Low |
| **Complex model** | Low | **High** |

#### 🧠 Worked example — bias & variance of a model

True value $f(x) = 100$. Retrain 4 times, predictions: 80, 82, 78, 80.
- Average $E[\hat f] = (80+82+78+80)/4 = 80$
- **Bias** $= 80 - 100 = -20$ (consistently underestimates → high bias)
- **Variance** = spread around 80: deviations $0, 2, -2, 0$; squared $0,4,4,0$; mean $= 2$ (small → low variance)

→ A **high-bias, low-variance** model (an underfitter — arrows tightly grouped but off-target).

#### ✏️ Practice example (made up — solved)

True value $f(x) = 50$. Retrain 4 times, predictions: **20, 80, 30, 70**.
- Average $E[\hat f] = (20+80+30+70)/4 = 200/4 = 50$
- **Bias** $= 50 - 50 = \mathbf{0}$ (on average, dead-on the bullseye!)
- **Variance** = spread around 50: deviations $-30, 30, -20, 20$; squared $900, 900, 400, 400$; mean $= 2600/4 = \mathbf{650}$ (huge)

→ A **low-bias, high-variance** model (an overfitter — arrows centred on the bullseye *on average* but wildly scattered, so any single prediction is unreliable). Contrast with the previous example: same idea, opposite failure.

---

## 10. Bias–Variance Trade-off & Learning Curves

### 🎯 Picture this — the dartboard

You **can't** make both bias and variance zero; reducing one usually raises the other. The art is **balancing** them.

```
                LOW Variance        HIGH Variance
              (shots clustered)    (shots scattered)

  HIGH Bias    ┌───────────┐        ┌───────────┐
 (off-centre)  │  ◯◯◯ off  │        │ × ◯× × ×  │
               │  UNDERFIT  │        │           │
               └───────────┘        └───────────┘

  LOW Bias     ┌───────────┐        ┌───────────┐
 (on-centre)   │  ⊗⊗⊗ tight│        │ × ×⊗× ×   │
               │ ★ BEST ★  │        │  OVERFIT  │
               └───────────┘        └───────────┘
```

| Quadrant | Meaning |
|---|---|
| High bias, Low variance | **Underfit** — consistently wrong, same spot |
| Low bias, High variance | **Overfit** — right on average but wildly scattered |
| Low bias, Low variance | ⭐ **Ideal** — accurate *and* consistent |

Mapped to the price/size curves: $\theta_0+\theta_1 x$ → underfit/high bias; $+\theta_2 x^2$ → just right; $+\dots+\theta_4 x^4$ → overfit/high variance.

### Learning curve

```
 Error
   │╲                              ╱ ← Testing Error (U-shape)
   │ ╲___              ___╱╱
   │     ╲___      ___╱    ← Training Error (keeps dropping)
   │         ╲────╱─────────
   │      Best Complexity (lowest test error)
   └───────────┊──────────────────► Model Complexity
```

- **Training error** always **decreases** as complexity rises (fits training data better and better).
- **Testing error** **decreases then increases** (U-shape) — rises once overfitting starts.
- ⭐ **Best = bottom of the testing-error U.**

> 🔑 Left of the dip = underfit (high bias). Right of the dip = overfit (high variance). Pick the model at the **minimum of the test error curve**.

---

## 11. Regularization (Ridge & Lasso)

> 🎯 **One-line idea:** a model overfits by cranking its weights huge to wiggle through every point. So we **add a fine for big weights** to the error score. Now the model only makes a weight big if it's *really worth it* → smoother, simpler model.
>
> Like a **word limit on an essay**: without a limit you ramble (overfit); the penalty forces you to keep only what matters.

### Starting point — RSS (plain error)

**RSS = Residual Sum of Squares = total squared error.** Residual = (actual − predicted), squared, summed. Plain regression only minimizes RSS:

$$\text{RSS} = \sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij} \right)^2$$

The two methods below = **RSS + a fine**; they differ only in how the fine is computed.

### 11.1 Ridge Regression (L2)

Add a penalty of **λ × (sum of squared weights)**:

$$\text{RSS} + \lambda \sum_{j=1}^{p}\beta_j^{2}$$

- **λ (lambda)** = how hard to penalize. λ=0 → plain regression (may overfit); λ huge → weights forced toward 0 (may underfit).
- Ridge **shrinks** weights toward zero but **rarely exactly zero**.

### 11.2 Lasso (L1)

Same idea, penalty uses **absolute values**:

$$\text{RSS} + \lambda \sum_{j=1}^{p}|\beta_j|$$

- Lasso can drive some weights **exactly to zero** → automatic **feature selection**.

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | $\lambda\sum \beta_j^2$ | $\lambda\sum |\beta_j|$ |
| Effect | Shrinks toward 0 (rarely =0) | Some become **exactly 0** |
| Bonus | — | Feature selection |

```
  λ too small ──────► λ just right ──────► λ too large
   OVERFIT             ★ BEST ★             UNDERFIT
  (weights huge)    (balanced)          (weights ≈ 0)
```

#### 🧠 Worked example — why the penalty discourages big weights

Two models fit equally well (same RSS = 10), λ = 1:
- Model A weights $\beta = (5, 5)$ → Ridge cost $= 10 + 1\times(25+25) = \mathbf{60}$
- Model B weights $\beta = (1, 1)$ → Ridge cost $= 10 + 1\times(1+1) = \mathbf{12}$

Both fit equally, but B has lower total cost → regularization **picks the simpler model B** (smaller weights = smoother = less overfitting).

#### ✏️ Practice example (made up — solved)

Same RSS = 8 for both models, but now use **λ = 2**, and compare **Ridge vs Lasso** on weights $\beta = (4, 0, 3)$ vs $\beta = (2, 0, 1)$.

*Model A, $\beta=(4,0,3)$:*
- Ridge: $8 + 2(4^2 + 0^2 + 3^2) = 8 + 2(16+0+9) = 8 + 50 = \mathbf{58}$
- Lasso: $8 + 2(|4|+|0|+|3|) = 8 + 2(7) = 8 + 14 = \mathbf{22}$

*Model B, $\beta=(2,0,1)$:*
- Ridge: $8 + 2(4 + 0 + 1) = 8 + 10 = \mathbf{18}$
- Lasso: $8 + 2(2+0+1) = 8 + 6 = \mathbf{14}$

→ Both penalties prefer the smaller-weight Model B (18 < 58, 14 < 22). Note the **middle weight is already 0** — Lasso is happy to keep it at exactly 0 (free feature removal), whereas Ridge would only ever shrink it *close* to 0.

> The penalty makes some weights ≈ 0, nudging an overfit model toward simpler. **Tuning λ correctly** lands it in the "just right" zone.

---

## 12. The Night-Before Recap

**Regression** = predict a continuous number (height → weight). Y = dependent (predict), X = independent (input).

**Simple linear regression line:** $\hat y = \theta_0 + \theta_1 x_1$ — best straight ruler through the cloud (θ₀ = intercept, θ₁ = slope).
$$\theta_1 = \frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}, \qquad \theta_0 = \bar y - \theta_1\bar x$$
Order: means → θ₁ → θ₀ → write equation → substitute. (Slide: $\hat y = 125.74 + 39x$. Practice: $\hat y = 15 + 16x$ → 5 hrs = 95 marks.)

**Model evaluation:** same data = memorized (bad). Train/test split = better. **K-fold CV = best** (avg of folds; slide avg(80,84,82,86)=83%; practice 5-fold = 75%).

**Multiple LR:** $\hat y = \theta^T X$ (the `1` in X pairs with θ₀). Each feature gives a weighted push.

**Overfit vs underfit** (the 3 exam-prep students): underfit = too simple (high train+test error); overfit = memorized noise (low train, high test). Fixes are opposites.

**Basis functions** = bend the paper, not the ruler: model = $\mathbf{w}^T\phi(\mathbf{x})$, linear in *weights*, fits curves.
- Polynomial $\phi_j=x^j$ → **GLOBAL**.
- Gaussian $\exp\{-(x-\mu)^2/2s^2\}$ → **LOCAL** (bump at $\mu$).
- Sigmoidal $\sigma((x-\mu)/s)$ → **LOCAL**.

**Bias & variance** (the archer): Bias $=\mathbb{E}[\hat f]-f$ (avg miss; simple→high). Variance $=\mathbb{E}[(\hat f-\mathbb{E}[\hat f])^2]$ (spread; complex→high). Can't kill both → trade-off; pick min of the test-error U-curve (dartboard).

**Regularization** = a fine for big weights (essay word limit) = RSS + penalty:
- **Ridge (L2):** $+\lambda\sum\beta_j^2$ → shrinks weights (rarely to 0).
- **Lasso (L1):** $+\lambda\sum|\beta_j|$ → some weights exactly 0 (feature selection).
- λ too small = overfit, too large = underfit.

---

*✅ Study guide for Lecture 2 (Linear Regression & Basics) complete.*
