# Lecture 1–3: Introduction to Machine Learning

> **Course:** BITS F464 — Machine Learning
> **Covers:** What ML is, AI/ML/DL relationship, why ML, Traditional Programming vs ML, the ML Recipe, all Types of Learning, and the Curse of Dimensionality.
>
> **How to read this:** Every concept is explained from scratch with intuition. Every formula has a fully worked numerical example so you *understand* it instead of memorizing it. Diagrams are drawn inline.

---

## Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Relationship between AI, ML and DL](#2-relationship-between-ai-ml-and-dl)
3. [Why do we require ML?](#3-why-do-we-require-ml)
4. [Traditional Programming vs ML](#4-traditional-programming-vs-ml)
5. [The Recipe of ML (6 components)](#5-the-recipe-of-ml-6-components)
6. [Types of Learning](#6-types-of-learning)
7. [Curse of Dimensionality](#7-curse-of-dimensionality)
8. [Quick Revision Sheet](#8-quick-revision-sheet)

---

## 1. What is Machine Learning?

### The core idea (in plain words)

Normally, a programmer writes **explicit rules**: "if email contains the word 'lottery', mark it as spam." This works only if a human can think of and write down every rule.

**Machine Learning flips this.** Instead of telling the computer *the rules*, we give it **lots of examples** (data) and let it **figure out the rules itself** by finding patterns. The more examples (experience) it sees, the better it gets — just like a student improving by practising more problems.

> 🔑 **One-line definition:** ML = algorithms that **iteratively learn patterns from data** to make predictions, **without being explicitly programmed** what to look for.

ML grew out of two older fields inside Artificial Intelligence:
- **Pattern recognition** — finding regularities in data (e.g., recognizing faces).
- **Computational learning theory** — the math of *how* and *how well* machines can learn.

### The famous definitions (memorize the Mitchell one — it's exam-favourite)

| Author (Year) | Definition | Easy meaning |
|---|---|---|
| **Arthur Samuel (1959)** | Field of study that gives computers the ability to learn **without being explicitly programmed**. | The original, simplest definition. |
| **Tom Mitchell (1998)** ⭐ | A program learns from **experience E** w.r.t. **task T** and **performance measure P**, if its performance at T (measured by P) **improves with E**. | The formal "T, P, E" definition. |
| **Ethem Alpaydın (2014)** | Programming computers to **optimize a performance criterion** using example data / past experience. | ML = optimization using data. |

#### 🧠 Worked example of Mitchell's T–P–E definition

Take a **spam filter**:

| Symbol | Meaning here |
|---|---|
| **T** (Task) | Classify an email as *spam* or *not spam* |
| **P** (Performance measure) | % of emails classified correctly (accuracy) |
| **E** (Experience) | Watching you mark emails as spam / not spam |

> If the filter's accuracy (P) on the task (T) **goes up** as it sees more of your marked emails (E) → by Mitchell's definition, **it is learning.** Always answer T–P–E questions by mapping each letter like this.

---

## 2. Relationship between AI, ML and DL

These three are **nested circles** — each is a subset of the bigger one.

```
   ┌─────────────────────────────────────────────┐
   │  ARTIFICIAL INTELLIGENCE (AI)                │
   │  Machine mimics human intelligence:          │
   │  predict, classify, learn, plan, reason,     │
   │  perceive.                                   │
   │                                              │
   │     ┌───────────────────────────────────┐    │
   │     │  MACHINE LEARNING (ML)            │    │
   │     │  Subset of AI. Uses math &        │    │
   │     │  statistics to learn from data    │    │
   │     │  and improve with experience.     │    │
   │     │                                   │    │
   │     │      ┌───────────────────────┐    │    │
   │     │      │  DEEP LEARNING (DL)   │    │    │
   │     │      │  Subset of ML. Uses   │    │    │
   │     │      │  neural networks for  │    │    │
   │     │      │  complex tasks: image,│    │    │
   │     │      │  audio, video.        │    │    │
   │     │      └───────────────────────┘    │    │
   │     └───────────────────────────────────┘    │
   └─────────────────────────────────────────────┘
```

| Term | What it is | Example |
|---|---|---|
| **AI** | Broadest — any machine that *acts smart* (even hand-coded rules). | A chess engine using fixed rules. |
| **ML** | A *subset* of AI that **learns from data** rather than being hard-coded. | Spam filter that improves with examples. |
| **DL** | A *subset* of ML using **multi-layered neural networks** for very complex data. | Face recognition, ChatGPT, speech-to-text. |

> 🔑 **Remember:** *All DL is ML, all ML is AI — but not all AI is ML.* (A rule-based system is AI but not ML.)

---

## 3. Why do we require ML?

We need ML for tasks **easy for humans but very hard to hand-code** as rules. You can't write "if-else" rules for "what does a cat look like" — but you *can* show 10,000 cat photos.

| Domain | Example tasks |
|---|---|
| **Vision** | Identify faces in a photo, detect objects in video/images |
| **Natural Language** | Translate Hindi → English, question answering, sentiment analysis |
| **Speech** | Recognize spoken words, speak sentences naturally |
| **Game playing** | Chess, Go, Dota |
| **Robotics** | Walking, jumping, displaying emotions |
| **Control** | Driving a car, navigating a maze |

> 💡 **Why hand-coding fails here:** the rules are either unknown to us, too many, or constantly changing. ML lets the data supply the rules.

---

## 4. Traditional Programming vs ML

This is the single most important conceptual contrast in the lecture. **The inputs and outputs are literally swapped.**

```
 TRADITIONAL PROGRAMMING                MACHINE LEARNING
 ───────────────────────                ────────────────

   Input  +  Program  ──►  Output         Input  +  Output  ──►  Program
   (data)   (rules you      (answer)       (data)   (answers    (rules the
            wrote)                                   you give)   machine learns)
```

| | Traditional Programming | Machine Learning |
|---|---|---|
| **You provide** | Data + the **Program (rules)** | Data + the **Output (answers/labels)** |
| **Computer produces** | The Output | The **Program (rules/model)** |
| **Who makes the rules?** | The human programmer (manual) | Learned automatically from data |

#### 🧠 Worked example (the slide's "late payment" example)

**Goal:** Predict whether a customer will pay their bill late.

- **Traditional way:** A human writes rules like *"if income < X and missed > 2 payments → late."* Hard, brittle, you must know the rule.
- **ML way:** You feed in:
  - **Input** = customer Demographics + past Bills
  - **Output** = "Paid late" or "Not late" (known historical answers)
  - The algorithm outputs a **"Late Payment Model"** (the program) that can now predict for *new* customers.

> 🔑 **Exam trick:** If the question gives you *answers* and asks for *rules* → it's ML. If it gives you *rules* and asks for *answers* → it's traditional programming.

---

## 5. The Recipe of ML (6 components)

Every ML system is built from **6 ingredients**. Memorize this list — it's a guaranteed exam question.

> **D-T-M-L-L-E:** **D**ata, **T**ask, **M**odel, **L**oss function, **L**earning algorithm, **E**valuation.

```
   ┌────────┐   ┌────────┐   ┌────────┐
   │ 1.DATA │ → │ 2.TASK │ → │ 3.MODEL│
   └────────┘   └────────┘   └────┬───┘
                                   ▼
   ┌────────────┐  ┌──────────────────┐  ┌──────────┐
   │6.EVALUATION│ ←│5.LEARNING ALGO   │ ←│ 4.LOSS   │
   │ accuracy   │  │ (minimize loss)  │  │ FUNCTION │
   └────────────┘  └──────────────────┘  └──────────┘
```

### 5.1 Data — *"the fuel of ML"*

- Can be **structured** (tables/spreadsheets) or **unstructured** (images, text, audio).
- **Sources:** open dataset repositories, crowd-sourcing marketplaces, or create your own.
- **Splitting of data** (very important):
  - **Training data** → used to *teach/train* the model.
  - **Test data** → *unseen* data used afterwards to *measure accuracy* honestly.

> 💡 **Why split?** Testing on the same data you trained on is like giving a student the exact exam questions to practise — you'd never know if they truly learned. Test data must be *new*.

### 5.2 Task — *what are we predicting?*

The **type of prediction/inference** being made, based on the question and available data.

| Task | What it does | Example |
|---|---|---|
| **Classification** | Assigns data to **categories** | Spam / not spam |
| **Clustering** | Groups data by **similarity** (no labels) | Customer segments |

### 5.3 Model — *the mathematical function*

> A **model** is just a **mathematical function** that maps **input → output**.

E.g., a model could be `output = a·x₁² + b·x₂³ + c·x₃²`. The letters `a, b, c` are **parameters** the machine must learn.

Related techniques mentioned (covered deeply in later lectures):
- **Bias–variance trade-off**, **Regularization**, **Overfitting** — all about making the model generalize well to new data, not just memorize training data.

### 5.4 Loss Function — *"how wrong is the model?"*

> A **loss function** measures **how good or bad** the model's predictions are. **Lower loss = better model.** Learning = minimizing this number.

Common loss functions: **Mean Squared Error (MSE)**, **Cross-Entropy**.

#### 📐 Formula: Mean Squared Error (MSE)

$$\mathcal{L} = \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i)\right)^2$$

| Symbol | Meaning |
|---|---|
| $n$ | number of data points |
| $y_i$ | the **true** value for point $i$ |
| $\hat{f}(x_i)$ | the value the **model predicted** for point $i$ |
| $\left(y_i - \hat{f}(x_i)\right)$ | the **error** for one point |
| squaring | makes all errors positive + punishes big errors more |

> *(Note: the "mean" version divides by $n$; the slide shows the **sum** version $\mathcal{L}_1 = \sum (y_i - \hat f(x_i))^2$ — both are minimized the same way.)*

#### 🧠 Worked example of MSE

Suppose a model predicts house prices (in lakhs):

| House | True price $y_i$ | Predicted $\hat{f}(x_i)$ | Error $(y_i-\hat f)$ | Squared error |
|---|---|---|---|---|
| 1 | 50 | 48 | 2 | 4 |
| 2 | 60 | 65 | −5 | 25 |
| 3 | 55 | 54 | 1 | 1 |

**Step 1 — sum the squared errors:** $4 + 25 + 1 = 30$
**Step 2 — Sum-of-squares loss** $\mathcal{L} = \mathbf{30}$
**Step 3 — Mean squared error** $= 30 / 3 = \mathbf{10}$

> 👉 The learning algorithm's whole job is to adjust `a, b, c` until this number `30` (or `10`) becomes as **small as possible**.

### 5.5 Learning Algorithm (Optimizer) — *finds the best parameters*

> Job: **find the parameter values** (like `a, b, c`) that make the **loss function minimum**.

Written mathematically as:

$$\min_{a,b,c} \; \mathcal{L} = \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i)\right)^2$$

This just reads: *"choose `a, b, c` that make the total error smallest."*

Common optimizers: **Gradient Descent**, **AdaGrad**.

#### 🧠 Worked mini-example (the slide's movie example)

Model: $\hat{f}(x) = a\,x_1^2 + b\,x_2^3 + c\,x_3^2$ predicting IMDB rating from (Budget, Box Office, Action-scene time).

The optimizer tries many values of `a, b, c`. The slide's table shows it settled on:

| a | b | c | Loss | Min Loss |
|---|---|---|---|---|
| 20.0 | 20.0 | −10.0 | 3.74 | 3.74 |

→ It found that **a=20, b=20, c=−10** gives the **smallest possible loss (3.74)**. That trained set of numbers *is* the final model.

### 5.6 Evaluation — *how accurate is the final model?*

After training, judge it on **test data** using metrics:
- **Accuracy**, **Precision**, **Recall**, **F1-score** (these are explained in detail in a later lecture).

> 🔑 **Summary chain:** Data feeds the Model; the Loss says how wrong it is; the Learning Algorithm tweaks parameters to shrink the loss; Evaluation gives the final report card.

---

## 6. Types of Learning

This is the biggest section. Here's the map first:

```
                 TYPES OF LEARNING
   ┌──────────────┬───────────────┬──────────────────┐
 Supervised   Unsupervised   Semi-supervised   Reinforcement
 (labels)     (no labels)    (few labels)      (rewards)

   + Bayesian   + Instance-based   + Active   + Deep Learning
   (probability)  (lazy/store)     (asks for   (neural nets)
                                    labels)
```

| Type | Data used | One-line idea |
|---|---|---|
| **Supervised** | Labelled | Learn input → known output mapping |
| **Unsupervised** | Unlabelled | Find hidden structure/groups |
| **Semi-supervised** | Few labelled + many unlabelled | Use the few labels to guide the rest |
| **Reinforcement** | No dataset; rewards | Learn by trial, error & feedback |
| **Bayesian** | Probabilities + data | Update belief using Bayes' theorem |
| **Instance-based** | Stored examples | "Lazy" — compare new point to stored ones |
| **Active** | Picks its own data | Ask a human to label the most useful points |
| **Deep** | Large data | Layered neural networks |

---

### 6.1 Supervised Learning ⭐

- Uses **labelled data** (every input comes with the correct answer).
- Learns a **mapping between input examples and the target variable**.

**Two sub-types:**

| Sub-type | Predicts | Example dataset |
|---|---|---|
| **Classification** | A **class label** (categories) | **MNIST**: image of handwritten digit → digit 0–9 |
| **Regression** | A **numerical value** | **Boston housing**: neighbourhood info → house price ($) |

> 💡 **Memory hook:** **C**lassification → **C**ategories. **R**egression → **R**eal numbers.

**Common algorithms:** K-Nearest Neighbour, Linear Regression, Decision Tree, Support Vector Machine (SVM).

```
   Labelled training data            New input
   (cat 🐱, dog 🐶, cat 🐱)  ──train──►  MODEL ──► "dog 🐶"
```

---

### 6.2 Unsupervised Learning

- Uses **unlabelled data** — only inputs, **no target/output**.
- Goal: **describe or extract relationships** in data (find hidden structure).

**Two sub-types:**

| Sub-type | What it does | Example |
|---|---|---|
| **Clustering** | Find natural **groups** in data | Market research, customer segmentation, image processing |
| **Density Estimation** | Summarize the **distribution** of data | Geographical & market data analysis |

**Common algorithms:** K-Means clustering, Hierarchical clustering, Kernel Density Estimation, Principal Component Analysis (PCA).

```
   Unlabelled points         After clustering
     • •    • •                (Group A)  (Group B)
    •  •   • •      ──►        ◉ ◉ ◉      ▲ ▲ ▲
     • •     •                 ◉ ◉        ▲ ▲
```

> ⚠️ **Slide typo note:** the slide says *"Two main types of **supervised** learning problems"* under Unsupervised — it should read **unsupervised**. Examiners sometimes test if you noticed.

---

### 6.3 Semi-Supervised Learning

- Uses **a few labelled examples + a large number of unlabelled examples**.
- It's *supervised learning where most labels are missing*. The few labels guide how to organize the huge unlabelled pile.
- **Cluster analysis**: partition data into homogeneous subgroups (similar together, different across groups).
- **Semi-supervised clustering** uses the *known* cluster info to classify the *unlabelled* data.

**Example:** A **text document classifier** — learns from a small set of labelled documents, then classifies a large amount of unlabelled documents.

**Common algorithms:** Co-training, Semi-supervised SVM.

> 💡 **Why it matters:** Labelling data is *expensive* (needs humans). Unlabelled data is *cheap and abundant*. Semi-supervised gets the best of both.

---

### 6.4 Reinforcement Learning (RL)

- An **agent** operates in an **environment** and must **learn from feedback** (rewards/punishments). No fixed dataset — it learns by *doing*.

**Four essential elements:**

| Element | Meaning | Game analogy (Mario) |
|---|---|---|
| **Agent** | The program you train | Mario (controlled by AI) |
| **Environment** | The world it acts in | The game level |
| **Action** | A move that changes the environment's state | Jump, move right |
| **Reward** | Evaluation of an action (+ or −) | +points for coins, − for dying |

```
        ┌──────── ACTION ────────►┐
   ┌─────────┐                ┌──────────────┐
   │  AGENT  │                │ ENVIRONMENT  │
   └─────────┘                └──────────────┘
        ▲                            │
        └──── STATE, REWARD ─────────┘
```

**Example:** Playing a game where the agent's goal is a **high score**, learning from rewards/punishments.
**Common algorithms:** Q-Learning, Markov Decision Process (MDP).

> 🔑 **Key difference from supervised:** No "correct answer" is given. The agent discovers good behaviour through **trial and error guided by rewards**.

---

### 6.5 Bayesian Learning

- Based on **Bayes' Theorem**.
- Each new training example **incrementally increases or decreases** the estimated probability that a hypothesis is correct (it doesn't throw a hypothesis away on one bad example — it's *flexible*).
- Combines **prior knowledge** with **observed data**. Prior knowledge is given by:
  1. A **prior probability** for each candidate hypothesis.
  2. A **probability distribution over observed data** for each hypothesis.

#### 📐 Bayes' Theorem

$$P(H \mid D) = \frac{P(D \mid H)\,\cdot\,P(H)}{P(D)}$$

| Symbol | Name | Meaning |
|---|---|---|
| $P(H)$ | **Prior** | Belief in hypothesis *before* seeing data |
| $P(D\mid H)$ | **Likelihood** | Probability of seeing this data *if* H were true |
| $P(H\mid D)$ | **Posterior** | Updated belief *after* seeing data (the answer) |
| $P(D)$ | **Evidence** | Total probability of the data (normalizer) |

#### 🧠 Worked example — the medical-test example from the slide

**Scenario:** A disease affects **1%** of people. A test is **99% accurate** for sick people (true positive), but has a **5% false-positive rate** for healthy people. **You test positive. What's the chance you're actually sick?**

Let:
- $H$ = "you have the disease". Prior $P(H) = 0.01$, so $P(\text{not }H)=0.99$.
- $D$ = "test is positive". Likelihoods: $P(D\mid H)=0.99$, $P(D\mid \text{not }H)=0.05$.

**Step 1 — total probability of a positive test, $P(D)$:**
$$P(D) = P(D|H)P(H) + P(D|\neg H)P(\neg H)$$
$$P(D) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594$$

**Step 2 — apply Bayes:**
$$P(H \mid D) = \frac{P(D|H)\,P(H)}{P(D)} = \frac{0.0099}{0.0594} \approx \mathbf{0.167}$$

> 😲 **Insight:** Even with a positive result on a "99% accurate" test, there's only a **~17% chance** you actually have the disease — because the disease is so rare (low prior). *This is exactly why Bayesian learning combines prior + data.* This counter-intuitive result is a classic exam question.

**Example use case:** Judging medical test accuracy considering how likely a person is to have a disease + test accuracy.
**Common algorithms:** Naïve Bayes Classifier, Gibbs Algorithm.

---

### 6.6 Instance-Based Learning ("Lazy Learning")

- **Stores the training examples** instead of learning an explicit function up front.
- Generalization is **postponed until a new instance must be classified** — that's why it's called **lazy learning**.
- When a new point arrives, it's compared to **stored examples** to decide its label.

**Key advantage:** Instead of one global function, it can estimate the target **locally and differently for each new point** → flexible, adapts to local structure.

**Example:** **Recommender systems** — "users who liked X also liked Y", so X and Y are probably similar.
**Common algorithms:** KNN (K-Nearest Neighbours), Locally Weighted Regression.

```
  Lazy (instance-based):  store all points → wait → compare on query
  Eager (e.g. linear reg): build one formula now → discard the data
```

> 💡 **Trade-off:** Lazy learners train instantly (just store data) but predict slowly (must compare against everything). Eager learners train slowly but predict fast.

---

### 6.7 Active Learning

- **Key idea:** an algorithm can reach **higher accuracy with fewer labelled examples** *if it gets to choose which data it learns from.*
- The **active learner asks queries**: it picks the most informative unlabelled instances and asks a **human annotator (oracle)** to label just those.

```
        learn a model
   ┌────────────────────►  ML MODEL
   │                          │ selects the most
 LABELED set  ◄──────────┐    │ useful queries
   (ℒ)        labels them │    ▼
                       ORACLE ◄── UNLABELED pool (𝒰)
                    (human annotator)
```

- **Why it matters:** unlabelled data is **abundant**, but labels are **difficult, time-consuming, or expensive**. Active learning spends the labelling budget *wisely*.
- **Slide's striking stat:** Active learning needed only **31 labelled points** to reach the **same accuracy** that passive (random) learning needed **174** points for. → ~5.6× more label-efficient.

**Example:** Gene expression and cancer classification.
**Common techniques:** Query Strategy Framework, analysis of active learning.

> 🔑 **Difference vs normal supervised:** Supervised learning is *passive* — it takes whatever labelled data it's given. Active learning is *picky* — it requests labels only for the points that will teach it the most.

---

### 6.8 Deep Learning (DL)

- A **subfield of ML**; powers the most human-like AI.
- Structures algorithms in **layers** to form an **Artificial Neural Network (ANN)** that learns and decides on its own.

```
  Input layer → Hidden layer 1 → Hidden layer 2 → ... → Output layer
    ●  ●  ●        ●  ●  ●           ●  ●            ●
    (each layer learns increasingly abstract features)
```

**Major types of ANN:**

| Network | Best for |
|---|---|
| **Feedforward NN** | Basic prediction (data flows one way) |
| **Recurrent NN (RNN)** | Sequences — text, speech, time series |
| **Convolutional NN (CNN)** | Images and video |

> 💡 **Why "deep"?** "Depth" = many hidden layers. Each layer builds on the previous one's features (edges → shapes → objects), letting it solve very complex problems automatically.

---

## 7. Curse of Dimensionality

### 7.1 What is a "dimension"?

> **Dimension = number of observables (features)** measured for each data point. E.g., if you record *age* and *income*, that's **2 dimensions**.

**When the number of dimensions increases, problems pile up:**
- ⏱️ More **time** to compute
- 💾 More **memory** to store inputs and intermediate results
- 🧩 More **complicated explanations** (e.g., regression with 100 parameters vs 2)
- 📊 **No simple visualization** (we can draw 2D/3D, not 10D)

> 🔑 **Definition:** The severe difficulties that arise in **high-dimensional spaces** are called the **Curse of Dimensionality**.

### 7.2 Example 1 — the oil/water/gas pipeline data

- **Dataset:** measurements of an oil + water + gas mixture from pipelines.
- **3 categories:** homogeneous (red), annular (green), laminar (blue).
- The slide plots **100 points using only 2 of the features** ($x_6$, $x_7$) — but in reality **each point is a 12-dimensional vector**.
- **Goal:** predict the class of a new point **×**.

### 7.3 Example continued — the intuition (slides 24–25)

Looking at where the new point **×** lands:
- Could be **Red** (surrounded by many red points)
- Could be **Green** (lots of green nearby)
- Unlikely **Blue** (few blue nearby)

> 🔑 **Core intuition:** *A point's class should be decided mostly by **nearby** training points, and less by **distant** ones.* (This single idea is the seed of the **K-Nearest Neighbours** algorithm!)

**Simple solution — the grid/cell method:**
1. Divide the input space into **regular cells**.
2. Predict a new point's class by **which cell it falls in** (e.g., majority vote of points in that cell).

```
   D=1: a line cut into 3 cells          → 3 cells
   ├──┼──┼──┤

   D=2: a grid 3×3                       → 9 cells
   ┌─┬─┬─┐
   ├─┼─┼─┤
   ├─┼─┼─┤
   └─┴─┴─┘

   D=3: a cube 3×3×3                     → 27 cells
   (cells explode as dimensions grow)
```

> ⚠️ **The curse strikes:** if each dimension is split into $k$ parts, total cells $= k^D$. The number of cells — and the training data needed to fill them — **grows exponentially** with dimension $D$.

### 7.4 Example 2 — polynomial coefficients blow up

A general order-3 polynomial in $D$ input variables:

$$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{i=1}^{D} w_i x_i + \sum_{i=1}^{D}\sum_{j=1}^{D} w_{ij} x_i x_j + \sum_{i=1}^{D}\sum_{j=1}^{D}\sum_{k=1}^{D} w_{ijk} x_i x_j x_k$$

**Reading this:** term 1 = constant; term 2 = linear (one sum); term 3 = quadratic (two nested sums = pairs); term 4 = cubic (three nested sums = triples).

- Number of coefficients grows **proportionally to $D^3$** for order 3.
- For polynomial of order $M$: number of coefficients $\approx D^M$.

#### 🧠 Worked example — feel the explosion

Order $M = 3$. Number of coefficients $\approx D^3$:

| $D$ (features) | Coefficients $\approx D^3$ |
|---|---|
| 2 | $2^3 = 8$ |
| 10 | $10^3 = 1{,}000$ |
| 100 | $100^3 = 1{,}000{,}000$ |

> 👉 Going from 10 to 100 features makes the model **1000× more complex**. Hence "limited practical utility because of complexity."

### 7.5 Remark + Solutions

- **More features ≠ better accuracy.** Adding features can actually make performance **worse**.
- Training examples required grows **exponentially** with dimensionality $d$ (i.e., $\propto k^d$).

```
 performance
   │      ╭─╮
   │     ╱   ╲___          ← accuracy rises, peaks,
   │    ╱        ╲────___     then FALLS as you add
   │   ╱                  ─── too many features
   └──────────────────────────► dimensionality
       (this is the "Hughes phenomenon")
```

**✅ Solutions to the curse:**
- **Dimensionality reduction** (e.g., PCA — keep only the most important combined features)
- **Feature extraction / selection** (drop useless features, build informative ones)

> 🔑 **Exam takeaway:** Curse of dimensionality = *as dimensions grow, data needed grows exponentially, computation/memory explode, and accuracy can drop.* Fix it by **reducing dimensions**.

---

## 8. Quick Revision Sheet

> Read this the night before the exam.

**Definitions**
- **ML** = learn patterns from data, no explicit programming.
- **Mitchell:** learns from **E** w.r.t. **T**, measured by **P**, if P on T improves with E.
- **AI ⊃ ML ⊃ DL** (nested; not all AI is ML).

**Traditional vs ML**
- Traditional: `Data + Program → Output`
- ML: `Data + Output → Program`

**ML Recipe (D-T-M-L-L-E):** Data, Task, Model, Loss function, Learning algorithm, Evaluation.

**Loss (MSE):** $\mathcal{L}=\sum (y_i-\hat f(x_i))^2$ — square the errors, add them; learning minimizes it.

**Types of learning**

| Type | Data | Key word |
|---|---|---|
| Supervised | labelled | classification (categories) / regression (numbers) |
| Unsupervised | unlabelled | clustering / density estimation |
| Semi-supervised | few labels + many unlabelled | cheap unlabelled + expensive labels |
| Reinforcement | rewards | agent, environment, action, reward |
| Bayesian | prior + data | Bayes' theorem, posterior |
| Instance-based | stored examples | lazy, KNN |
| Active | self-chosen | asks oracle for best labels |
| Deep | big data | layered neural networks (FFN/RNN/CNN) |

**Bayes:** $P(H|D)=\dfrac{P(D|H)P(H)}{P(D)}$ — rare disease + positive test ⇒ still low probability (the ~17% example).

**Curse of dimensionality:** cells/coefficients/data needed grow **exponentially** ($k^D$, $D^M$) with dimensions; more features can hurt accuracy. **Fix:** dimensionality reduction & feature extraction.

---

*✅ Study guide for Lecture 1–3 complete. Send the next PPT whenever ready.*
