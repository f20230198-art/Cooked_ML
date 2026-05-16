# Lecture 2: Linear Regression & Basics

> **Course:** BITS F464 — Machine Learning
> **Covers:** Regression intro, Simple & Multiple Linear Regression (with full worked θ₀/θ₁ calculation), Model Evaluation (train/test split, k-fold CV), Overfitting/Underfitting, Linear Basis Function models (polynomial, Gaussian, sigmoidal), Bias–Variance trade-off, and Regularization (Ridge L2 / Lasso L1).
>
> **How to read this:** Every formula has a fully worked numerical example. The slide-7 calculation (the one your teacher worked out by hand) is broken into baby steps — that's the most exam-likely problem in this lecture.

---

## Table of Contents

1. [What is Regression?](#1-what-is-regression)
2. [Types of Regression Models](#2-types-of-regression-models)
3. [Simple Linear Regression](#3-simple-linear-regression)
4. [⭐ Worked Example: Finding θ₀ and θ₁ (slide 7)](#4--worked-example-finding-θ₀-and-θ₁-the-most-important-problem)
5. [Model Evaluation](#5-model-evaluation)
6. [Multiple Linear Regression](#6-multiple-linear-regression)
7. [Overfitting and Underfitting](#7-overfitting-and-underfitting)
8. [Linear Basis Function Models](#8-linear-basis-function-models)
9. [Bias and Variance](#9-bias-and-variance)
10. [Bias–Variance Trade-off & Learning Curves](#10-biasvariance-trade-off--learning-curves)
11. [Regularization (Ridge & Lasso)](#11-regularization-ridge--lasso)
12. [Quick Revision Sheet](#12-quick-revision-sheet)

---

## 1. What is Regression?

> **Regression = predicting a continuous (numerical) value.** (e.g., price, temperature, CO₂ emission — *not* a category.)

Two key vocabulary words you MUST know:

| Term | Other names | Meaning | In the slide's car example |
|---|---|---|---|
| **Dependent variable (Y)** | target, response, outcome | The thing we **want to predict** | `CO2EMISSIONS` |
| **Independent variable (X)** | explanatory, predictor, feature | The **cause / input** we use to predict | `ENGINESIZE`, `CYLINDERS`, `FUELCONSUMPTION` |

> 💡 **Memory hook:** Y **dep**ends on X. X is **indep**endent (it stands alone, it's the input).

A **regression model** relates **Y to a function of X**:  $Y = f(X)$

**Slide's example:** Given historical car data (engine size, cylinders, fuel consumption, CO₂), build a model so that for a **new car** you can predict its **expected CO₂**. Row 9 in the table has `?` for CO₂ — that's exactly what the model fills in.

---

## 2. Types of Regression Models

```
                    REGRESSION MODELS
            ┌──────────────────┴──────────────────┐
      LINEAR Regression                  NON-LINEAR Regression
      (fits a straight line)             (fits a curve)
      ┌────────┴────────┐                ┌────────┴────────┐
   Simple           Multiple          Simple           Multiple
  (1 input)        (many inputs)     (1 input)        (many inputs)
```

| Type | Shape it fits | Slide example |
|---|---|---|
| **Linear** | Straight line / flat plane | `%Fat = −23.19 + 3.286·BMI − 0.04·BMI²` (BMI → body fat) |
| **Non-linear** | Curve (S-shape, etc.) | `Mobility = 1288 + 1491·DensityLn + 583·DensityLn² …` |
| **Simple** | Exactly **1** independent variable | Engine size → CO₂ |
| **Multiple** | **Many** independent variables | Engine size + cylinders + fuel → CO₂ |

---

## 3. Simple Linear Regression

> **Simple linear regression** = relationship between **one dependent** variable and **one independent** variable, fitted with a **straight line**.

A scatter plot shows how the two variables move together. Linear regression **fits a line through that cloud of points**.

```
 Emission
   │              •  •   ╱  ← fitted line
   │           •  • ╱ •
   │        • •╱ •  •
   │     •╱ • •           "best straight line
   │  •╱  •                through the data"
   └────────────────────► Engine size
```

### The line equation (same line, different notations)

All three of these are **the exact same equation** — different books use different letters. Don't let it confuse you:

| Notation | Slope | Intercept |
|---|---|---|
| $y = mx + c$ | $m$ | $c$ |
| $y = ax + b$ | $a$ | $b$ |
| $y = w_1 x + w_0$ | $w_1$ | $w_0$ |
| **$\hat{y} = \theta_0 + \theta_1 x_1$** (used in this course) | $\theta_1$ | $\theta_0$ |

### Anatomy of $\hat{y} = \theta_0 + \theta_1 x_1$

```
   ŷ      =   θ₀     +    θ₁    ·   x₁
   ▲           ▲           ▲         ▲
   │           │           │         │
 predicted   bias       slope /    predictor /
 value /    coefficient  prediction independent
 response   (intercept)  coefficient variable
 (dependent)
```

- **$\theta_0$ = bias coefficient (the intercept)** — where the line crosses the y-axis when $x=0$.
- **$\theta_1$ = the coefficient used for prediction (the slope)** — how much $y$ changes when $x$ goes up by 1.
- Once we **learn** $\theta_0$ and $\theta_1$ from data, we can **predict Y for any new X**.

---

## 4. ⭐ Worked Example: Finding θ₀ and θ₁ (the most important problem)

This is the slide-7 hand calculation. **This type of question is extremely likely in your exam.** Memorize the *procedure*, not numbers.

### The two formulas

**Slope:**
$$\theta_1 = \frac{\sum_{i=1}^{s}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{s}(x_i - \bar{x})^2}$$

**Intercept:**
$$\theta_0 = \bar{y} - \theta_1 \bar{x}$$

| Symbol | Meaning |
|---|---|
| $x_i$ | $i$-th input value (here: engine size) |
| $y_i$ | $i$-th output value (here: CO₂ emission) |
| $\bar{x}$ | **mean** (average) of all $x$ values |
| $\bar{y}$ | **mean** of all $y$ values |
| $s$ | number of data points |

> 💡 **What the slope formula *means*:** the top measures *"do x and y rise together?"* (covariance). The bottom measures *"how spread out is x?"* (variance of x). Slope = how strongly y moves per unit of x.

### Step-by-step using the slide's data

Data (engine size $x$ → CO₂ $y$): the 9 points are
$x$ = (2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7)
$y$ = (196, 221, 136, 255, 244, 230, 232, 255, 267)

**STEP 1 — Compute the means.**
$$\bar{x} = \frac{2.0 + 2.4 + 1.5 + \dots}{9} = 3.03$$
$$\bar{y} = \frac{196 + 221 + 136 + \dots}{9} = 226.22$$

**STEP 2 — Plug into the slope formula** (numerator = sum of $(x_i-\bar x)(y_i-\bar y)$; denominator = sum of $(x_i-\bar x)^2$):
$$\theta_1 = \frac{(2.0-3.03)(196-226.22) + (2.4-3.03)(221-226.22) + \dots}{(2.0-3.03)^2 + (2.4-3.03)^2 + \dots}$$

Let's verify the first term so you see *exactly* how to compute it:
- $(2.0 - 3.03) = -1.03$
- $(196 - 226.22) = -30.22$
- product $= (-1.03)(-30.22) = +31.13$ → goes into the **numerator**
- $(2.0 - 3.03)^2 = (-1.03)^2 = 1.06$ → goes into the **denominator**

You repeat this for all 9 rows, add up the numerator, add up the denominator, divide:
$$\boxed{\theta_1 = 39}$$

**STEP 3 — Compute the intercept** using $\theta_0 = \bar{y} - \theta_1\bar{x}$:
$$\theta_0 = 226.22 - 39 \times 3.03 = 226.22 - 118.17 = 125.74$$
$$\boxed{\theta_0 = 125.74}$$

**STEP 4 — Write the final model:**
$$\boxed{\hat{y} = 125.74 + 39\,x_1}$$

**STEP 5 — Use it to predict.** Row 9 had engine size $x = 2.4$ and CO₂ = `?`:
$$\hat{y} = 125.74 + 39 \times 2.4 = 125.74 + 93.6 = \mathbf{219.34}$$

> ✅ **Exam recipe (memorize this order):** ① means $\bar x,\bar y$ → ② slope $\theta_1$ → ③ intercept $\theta_0 = \bar y - \theta_1\bar x$ → ④ write $\hat y = \theta_0+\theta_1 x$ → ⑤ substitute the new $x$ to predict.

---

## 5. Model Evaluation

After building a model, you must **check how good it is**. The lecture gives **3 approaches**.

### 5.1 Train and Test on the SAME dataset ❌ (bad)

Train the model on all data, then test on that *same* data.
- ⚠️ Gives **high training accuracy** but **low out-of-sample accuracy** → it **memorized**, didn't learn.
- This is the classic cause of **overfitting**.

| Term | Definition |
|---|---|
| **Training accuracy** | % correct predictions on data the model **was trained on** |
| **Out-of-sample accuracy** | % correct on data the model **has NOT seen** (the honest number) |

> 💡 Like practising with the exact exam paper — looks like 100%, proves nothing.

### 5.2 Train/Test Split ✅ (better)

Split data into two **mutually exclusive** groups: a training set and a separate testing set.

```
   ENTIRE DATASET
   ● ● ● ● ● ● ● ●        split
   ● ● ● ● ● ● ● ●   ──────────►   TRAINING SET        TESTING SET
   ● ● ● ● ● ● ● ●                 ● ● ● ● ●            ● ●
   ● ● ● ● ● ● ● ●                 ● ● ● ● ●  (train)   ● ●  (test, unseen)
```

- ✅ **Truly out-of-sample testing** → more realistic accuracy.
- After evaluating, you can re-train on the testing set too (use all data for the final model).
- ⚠️ Result is **highly dependent on *which* points landed in train vs test** (luck of the split).

### 5.3 K-Fold Cross-Validation ✅✅ (best — most consistent)

Fixes the "unlucky split" problem. Split data into **k folds**. Each round: use one fold as test, the rest as train. Repeat so **every fold gets to be the test set once**. Average all the accuracies.

```
 Round 1:  [TEST ] [train] [train] [train]  → 80%
 Round 2:  [train] [TEST ] [train] [train]  → 84%
 Round 3:  [train] [train] [TEST ] [train]  → 82%
 Round 4:  [train] [train] [train] [TEST ]  → 86%

 Final Accuracy = avg(80 + 84 + 82 + 86) = 83%
```

#### 🧠 Worked example (the slide's numbers)

4 folds give accuracies **80%, 84%, 82%, 86%**:
$$\text{Accuracy} = \frac{80 + 84 + 82 + 86}{4} = \frac{332}{4} = \mathbf{83\%}$$

> ✅ **Why best:** every data point is used for *both* training and testing (in different rounds) → **more consistent, reliable** out-of-sample accuracy, not dependent on one lucky split.

---

## 6. Multiple Linear Regression

**Use it when there is more than one independent variable.** Two typical questions it answers:
- **Effectiveness:** *"Do revision time, anxiety, attendance & gender affect exam performance?"*
- **Impact of change:** *"How much does blood pressure change per unit change in BMI?"*

### The equation

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

This can be written compactly using vectors:

$$\hat{y} = \theta^T X$$

where:
$$\theta^T = [\theta_0,\ \theta_1,\ \theta_2,\ \dots] \qquad X = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \end{bmatrix}$$

> 💡 **Why the `1` on top of X?** It pairs with $\theta_0$ so the intercept fits neatly into the same multiplication. $\theta^T X = \theta_0\cdot 1 + \theta_1 x_1 + \theta_2 x_2 + \dots$ — exactly the long form. **The `T` means "transpose"** (turn the column of θ's into a row so it can multiply the column X).

#### 🧠 Mini worked example

Say a trained model is $\hat y = 50 + 30\,x_1 + 15\,x_2$ predicting CO₂ from engine size $x_1$ and cylinders $x_2$. For a car with engine size $2.0$ and $4$ cylinders:
$$\hat y = 50 + 30(2.0) + 15(4) = 50 + 60 + 60 = \mathbf{170}$$

---

## 7. Overfitting and Underfitting

> **The central challenge of ML:** perform well on **new, unseen** inputs — not just the training data. This is called **generalization**.

- **Generalization error** = **test error** (error on unseen data). We want this **low**.
- Test error is usually **≥** training error (model was built from training data, so it's "easier" on training data).
- **Two goals** (and their two failures):
  1. Make **training error small** → fail here = **underfitting**
  2. Make the **gap between train & test error small** → fail here = **overfitting**

### The two failure modes

| | **Underfitting** | **Overfitting** |
|---|---|---|
| **Definition** | Learns **too little** of the true pattern | Fits the data **too well** (memorizes noise) |
| **Causes** | Features don't capture concept; too much **bias**; too little search | Noisy/irrelevant features; model too powerful/sensitive; too much search |
| **Train error** | High | Very low |
| **Test error** | High | High |
| **Model type** | Too **simple** | Too **complex** |

```
  UNDERFIT (too simple)     GOOD FIT (just right)     OVERFIT (too complex)
  Price │   ___────         Price │    ╱─────         Price │  ╱╲    ╱╲╱
        │ ╱ × ×  ×                │  ╱× ×  ×                │ ×╲╱ ╲× ╱  ×
        │× ×   ×                  │ ×  ×  ×                 │× wiggles through
        │ straight line           │ smooth curve            │ EVERY point
        └──────► Size             └──────► Size             └──────► Size
   θ₀+θ₁x                    θ₀+θ₁x+θ₂x²          θ₀+θ₁x+θ₂x²+θ₃x³+θ₄x⁴
```

The overfit curve passes through every training point perfectly but **wiggles crazily** — it will predict terribly on a new point.

### How to fix each one (slide 15 — exam favourite list)

**To reduce OVERFITTING:**
- **Cross-validation** (train/test split or k-fold)
- **Train with more data** (doesn't always work)
- **Reduce features**
- **Early stopping** (stop training before it memorizes)
- **Regularization** (add a penalty term — see §11)

**To reduce UNDERFITTING:**
- **Increase model complexity**
- **Add more features / feature engineering**
- **Remove noise** from data
- **Train longer**

> 🔑 Notice: the fixes are **opposites**. Overfit → simplify/penalize. Underfit → add complexity/features. Many exam questions just ask "name 3 ways to reduce overfitting."

---

## 8. Linear Basis Function Models

This section generalizes linear regression so it can fit **curves**, not just straight lines.

### 8.1 The plain linear model (slide 16)

$$y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \dots + w_D x_D$$

where $\mathbf{x} = (x_1, \dots, x_D)^T$ are the $D$ input variables.

- It is **linear in both the parameters ($w$) AND the input variables ($x$)**.
- ⚠️ **Limitation:** can only fit a **straight line / flat plane**. Can't model curves.

### 8.2 The basis function trick (slide 17)

**Idea:** instead of using raw $x$, pass it through fixed **nonlinear functions** $\phi_j(x)$ first, then combine those *linearly*.

$$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j(\mathbf{x})$$

| Symbol | Meaning |
|---|---|
| $\phi_j(x)$ | a **basis function** (a fixed nonlinear transform of the input)* |
| $M$ | total number of parameters in the model |
| $w_0$ | the **bias parameter** (fixed offset) |

> *The slide calls $\phi_j$ a "bias function" — it actually means **basis function**. (Minor slide typo. Only $w_0$ is the *bias parameter*.)

By defining a dummy $\phi_0(\mathbf{x}) = 1$, it compresses to the clean vector form:

$$y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$$

where $\mathbf{w} = (w_0,\dots,w_{M-1})^T$ and $\boldsymbol{\phi} = (\phi_0,\dots,\phi_{M-1})^T$.

> 🔑 **The big idea:** the model is still **linear in the weights $w$** (so it's easy to train), but because $\phi$ is nonlinear, it can fit **curves**. "Linear model" refers to linear in *weights*, not in *x*.

### 8.3 Three types of basis functions

#### (a) Polynomial basis — $\phi_j(x) = x^j$

Gives **polynomial regression**: $\phi_0=1,\ \phi_1=x,\ \phi_2=x^2,\ \phi_3=x^3,\dots$

- ⚠️ **Limitation: they are GLOBAL.** A change in one region of the input affects the curve **everywhere** (the whole polynomial shifts).
- **Fix:** split the input space into regions, use a different polynomial in each → **spline functions** (polynomials joined together; flexible but complex).

#### (b) Gaussian basis — LOCAL

$$\phi_j(x) = \exp\left\{ -\frac{(x - \mu_j)^2}{2s^2} \right\}$$

| Symbol | Controls |
|---|---|
| $\mu_j$ | **location** (where the bump is centred) |
| $s$ | **scale / width** of the bump |

- These are **LOCAL**: a small change in $x$ only affects **nearby** basis functions (a bell-curve "bump" — far away it's ≈0).

```
  φ(x)  ╱╲   ╱╲   ╱╲   ╱╲    ← each bump is local;
        ╱  ╲ ╱  ╲ ╱  ╲ ╱  ╲    moving x only wakes up
       ╱    X    X    X    ╲   the nearest bumps
      ─────────────────────► x
       μ₁   μ₂   μ₃   μ₄
```

#### (c) Sigmoidal basis — LOCAL

$$\phi_j(x) = \sigma\!\left( \frac{x - \mu_j}{s} \right) \qquad \text{where}\quad \sigma(a) = \frac{1}{1 + e^{-a}}$$

- $\sigma(a)$ is the **sigmoid (logistic) function** — an S-shaped curve from 0 to 1.
- $\mu_j$ controls **location**, $s$ controls **scale (slope/steepness)**.
- Also **LOCAL** — small change in $x$ only affects nearby basis functions.

#### 🧠 Worked example — compute a sigmoid value

Let $\mu_j = 0$, $s = 1$, and input $x = 2$:
$$a = \frac{x - \mu_j}{s} = \frac{2 - 0}{1} = 2$$
$$\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135} = \frac{1}{1.135} \approx \mathbf{0.88}$$

> Sanity check: at $x=\mu$ (i.e. $a=0$): $\sigma(0)=\frac{1}{1+1}=0.5$ — the curve's midpoint. As $x\to\infty$, $\sigma\to1$; as $x\to-\infty$, $\sigma\to0$. That's the S-shape.

> 🔑 **Polynomial = global** (one change affects everywhere). **Gaussian & Sigmoidal = local** (one change affects only nearby region). This global-vs-local contrast is a classic exam question.

---

## 9. Bias and Variance

These two define *why* models underfit or overfit.

### 9.1 Bias

> **Bias** = the model's inability to capture the true relationship. It's the difference between the **average prediction** and the **correct value**.

Let $f(x)$ = the true model, $\hat{f}(x)$ = our estimate:

$$\text{Bias}\big(\hat{f}(x)\big) = \mathbb{E}\big[\hat{f}(x)\big] - f(x)$$

(Read $\mathbb{E}[\cdot]$ as "the average/expected value of".)

- **High bias** → model pays too little attention to training data, **oversimplifies** → **UNDERFITTING**.
- **Simple model → high bias. Complex model → low bias.**

### 9.2 Variance

> **Variance** = how much the model's prediction **wobbles** if you retrain it on different samples of data. It measures the spread.

$$\text{Variance}\big(\hat{f}(x)\big) = \mathbb{E}\Big[\big(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\big)^2\Big]$$

- **High variance** → model pays too much attention to training data (incl. noise), **doesn't generalize** → great on train, bad on test → **OVERFITTING**.
- **Simple model → low variance. Complex model → high variance.**

| | Bias | Variance |
|---|---|---|
| **Simple model** | **High** | Low |
| **Complex model** | Low | **High** |
| **Causes** | Underfitting | Overfitting |

#### 🧠 Worked example — bias of a model

True value $f(x) = 100$. We retrain the model 4 times on different samples and it predicts: 80, 82, 78, 80.
- Average prediction $\mathbb{E}[\hat f(x)] = (80+82+78+80)/4 = 80$
- **Bias** $= 80 - 100 = \mathbf{-20}$ (it consistently underestimates → high bias)
- **Variance** = spread around 80: deviations are $0, 2, -2, 0$; squared = $0,4,4,0$; mean = $\mathbf{2}$ (small → low variance)

→ This is a **high-bias, low-variance** model (an underfitter).

---

## 10. Bias–Variance Trade-off & Learning Curves

You **cannot** make both bias and variance zero. Reducing one usually increases the other. The art is **balancing** them.

### The dartboard analogy (slide 23 — very famous)

```
                LOW Variance        HIGH Variance
              (shots clustered)    (shots scattered)

  HIGH Bias    ┌───────────┐        ┌───────────┐
 (off-centre)  │     ◯     │        │  ×    ×   │
               │    ◯◯◯←×× │        │ × ◯ ×  ×  │   × = shots
               │     ◯  off │        │  × ◯× ×   │   ◯ = bullseye(truth)
               │  UNDERFIT  │        │           │
               └───────────┘        └───────────┘

  LOW Bias     ┌───────────┐        ┌───────────┐
 (on-centre)   │     ⊗     │        │ ×  × ×    │
               │   ⊗⊗⊗     │        │×  ⊗  ×  × │
               │  tight on │        │  × × ×    │
               │  bullseye │        │  OVERFIT  │
               │ ★ BEST ★  │        │           │
               └───────────┘        └───────────┘
```

| Quadrant | Meaning |
|---|---|
| High bias, Low variance | **Underfit** — consistently wrong, same wrong spot |
| Low bias, High variance | **Overfit** — right on average but wildly scattered |
| Low bias, Low variance | ⭐ **Ideal** — accurate and consistent |

Mapped to the price/size curves:
- $\theta_0+\theta_1 x$ → **"Underfit" / High bias**
- $\theta_0+\theta_1 x+\theta_2 x^2$ → **"Just right"**
- $\theta_0+\dots+\theta_4 x^4$ → **"Overfit" / High variance**

### Learning curve (slide 24)

```
 Error
   │╲                              ╱ ← Testing Error
   │ ╲                           ╱     (goes UP if too complex)
   │  ╲___              ___╱╱
   │      ╲___      ___╱    ← Training Error
   │          ╲────╱────────────  (keeps going DOWN)
   │           ┊
   │      Best Complexity (lowest test error)
   └───────────┊──────────────────► Model Complexity
```

- **Training error** always **decreases** as the model gets more complex (it fits training data better and better).
- **Testing error** **decreases then increases** (U-shape) — it goes back up once the model starts overfitting.
- ⭐ **Best complexity = the bottom of the testing-error U** — the sweet spot balancing bias and variance.

> 🔑 **Exam takeaway:** Left of the dip = underfit (high bias). Right of the dip = overfit (high variance). Pick the model at the **minimum of the test error curve**.

---

## 11. Regularization (Ridge & Lasso)

> **Regularization** = a technique to reduce overfitting by adding an **extra penalty term** to the error function. It **discourages overly complex models** and **reduces variance**.

**Intuition:** Overfitting happens when weights ($\beta_j$) get huge to wiggle through every point. Regularization **punishes large weights**, forcing the model to stay simpler/smoother.

### Starting point — RSS (Residual Sum of Squares)

For a multiple linear regression $Y \approx \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$, the basic loss is:

$$\text{RSS} = \sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij} \right)^2$$

This is just "sum of squared errors" (how wrong the model is). Plain regression minimizes only this.

### 11.1 Ridge Regression (L2)

Add a penalty equal to **λ × (sum of squared weights)**:

$$\underbrace{\sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij} \right)^2}_{\text{RSS}} + \lambda \sum_{j=1}^{p}\beta_j^{2} \;=\; \text{RSS} + \lambda \sum_{j=1}^{p}\beta_j^{2}$$

- **L2** penalty = sum of **squared** coefficients $\sum \beta_j^2$.
- **λ (lambda) = tuning parameter**: how hard to penalize complexity.
  - λ = 0 → no penalty (plain regression, may overfit)
  - λ very large → weights forced toward 0 (may underfit)
- Ridge **shrinks** weights toward zero but **rarely makes them exactly zero**.

### 11.2 Lasso (L1)

Same idea but penalty uses **absolute values** instead of squares:

$$\sum_{i=1}^{n}\left( y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p}|\beta_j| \;=\; \text{RSS} + \lambda \sum_{j=1}^{p}|\beta_j|$$

- **L1** penalty = sum of **absolute** coefficients $\sum |\beta_j|$.
- Lasso can drive some weights **exactly to zero** → automatically **removes useless features** (feature selection).

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | $\lambda\sum \beta_j^2$ (squared) | $\lambda\sum |\beta_j|$ (absolute) |
| Effect on weights | Shrinks toward 0 (rarely =0) | Some become **exactly 0** |
| Bonus | — | Feature selection |

### How regularization prevents overfitting

> The penalty term makes some weights **≈ 0**, which **converts an overfit model toward an underfit one**. **Correctly tuning λ** lands it in the **"just right"** zone.

```
  λ too small ──────► λ just right ──────► λ too large
   OVERFIT             ★ BEST ★             UNDERFIT
  (weights huge)    (balanced)          (weights ≈ 0)
```

#### 🧠 Worked example — why the penalty discourages big weights

Suppose two models fit the training data **equally well** (same RSS = 10), with λ = 1:
- Model A weights $\beta = (5, 5)$ → Ridge cost $= 10 + 1\times(5^2+5^2) = 10 + 50 = \mathbf{60}$
- Model B weights $\beta = (1, 1)$ → Ridge cost $= 10 + 1\times(1^2+1^2) = 10 + 2 = \mathbf{12}$

Even though both fit equally, **Model B has lower total cost**, so regularization **picks the simpler model B** (smaller weights = smoother = less overfitting). That's the whole mechanism.

---

## 12. Quick Revision Sheet

> Night-before-exam cheat sheet.

**Regression** = predict continuous value. Y = dependent (predict), X = independent (cause).

**Simple linear regression line:** $\hat y = \theta_0 + \theta_1 x_1$ (θ₀ = bias/intercept, θ₁ = slope).

**Finding the line (memorize procedure):**
$$\theta_1 = \frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}, \qquad \theta_0 = \bar y - \theta_1\bar x$$
Order: means → θ₁ → θ₀ → write equation → substitute new x. (Slide example answer: $\hat y = 125.74 + 39x$.)

**Model evaluation:**
- Same data = bad (overfits). Train/Test split = better. **K-fold CV = best** (avg of folds; slide: avg(80,84,82,86)=83%).

**Multiple LR:** $\hat y = \theta^T X$ (the `1` in X pairs with θ₀).

**Overfit vs Underfit:**

| | Underfit | Overfit |
|---|---|---|
| Model | Too simple | Too complex |
| Train err | High | Low |
| Test err | High | High |
| Bias | High | Low |
| Variance | Low | High |
| Fix | add complexity/features | regularize, more data, fewer features, early stop, CV |

**Basis functions:** model = $\mathbf{w}^T\phi(\mathbf{x})$, linear in *weights* (fits curves).
- Polynomial $\phi_j=x^j$ → **GLOBAL**.
- Gaussian $\exp\{-(x-\mu)^2/2s^2\}$ → **LOCAL**.
- Sigmoidal $\sigma((x-\mu)/s)$, $\sigma(a)=1/(1+e^{-a})$ → **LOCAL**.

**Bias** $=\mathbb{E}[\hat f]-f$ (simple model → high). **Variance** $=\mathbb{E}[(\hat f-\mathbb{E}[\hat f])^2]$ (complex model → high). Can't minimize both → **trade-off**; pick min of test-error U-curve.

**Regularization** = RSS + penalty:
- **Ridge (L2):** $+\lambda\sum\beta_j^2$ → shrinks weights.
- **Lasso (L1):** $+\lambda\sum|\beta_j|$ → some weights = 0 (feature selection).
- λ controls strength: too small = overfit, too large = underfit.

---

*✅ Study guide for Lecture 2 (Linear Regression & Basics) complete. Send the next PPT whenever ready.*
