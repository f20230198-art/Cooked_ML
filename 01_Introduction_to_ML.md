# Lecture 1–3: Introduction to Machine Learning

> **Course:** BITS F464 — Machine Learning
> **Covers:** What ML is, the AI/ML/DL relationship, why we need ML, Traditional Programming vs ML, the ML Recipe, all the Types of Learning, and the Curse of Dimensionality.
>
> **How this chapter is written:** every idea opens with an everyday-life story (no math), then connects that story to *why* we do the thing, then shows the method/formula with every symbol explained in the story's language. Every original worked example is kept, and important formulas get an **extra made-up example, fully solved**.

---

## Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Relationship between AI, ML and DL](#2-relationship-between-ai-ml-and-dl)
3. [Why do we require ML?](#3-why-do-we-require-ml)
4. [Traditional Programming vs ML](#4-traditional-programming-vs-ml)
5. [The Recipe of ML (6 components)](#5-the-recipe-of-ml-6-components)
6. [Types of Learning](#6-types-of-learning)
7. [Curse of Dimensionality](#7-curse-of-dimensionality)
8. [The Night-Before Recap](#8-the-night-before-recap)

---

## 1. What is Machine Learning?

### 🎯 Picture this

Imagine teaching a child to recognize a dog. You **don't** hand them a rulebook: "if it has four legs AND fur AND a tail AND barks…" — that list would be endless and still fail (cats have four legs and fur too). Instead, you just **point at lots of dogs**: "that's a dog… that's a dog… that's a cat, not a dog." After enough examples, the child *figures out the pattern themselves* and can spot a dog they've never seen before.

That switch — from **writing the rules** to **showing examples and letting the learner find the rules** — is the entire idea of Machine Learning.

### 🔗 Why this matters

Normally a programmer writes **explicit rules**: "if the email contains the word 'lottery', mark it as spam." That only works if a human can think of and write down *every* rule. For "what does a dog look like," nobody can.

**Machine Learning flips it.** Give the computer **lots of examples (data)** and let it **discover the rules itself** by finding patterns. More examples (more experience) → better performance — exactly like a student who improves by practising more problems.

> 🔑 **One-line definition:** ML = algorithms that **iteratively learn patterns from data** to make predictions, **without being explicitly programmed** what to look for.

ML grew out of two older ideas inside Artificial Intelligence:
- **Pattern recognition** — finding regularities in data (e.g., recognizing faces).
- **Computational learning theory** — the math of *how* and *how well* machines can learn.

### 📐 The famous definitions

| Author (Year) | Definition | Easy meaning |
|---|---|---|
| **Arthur Samuel (1959)** | Field of study that gives computers the ability to learn **without being explicitly programmed**. | The original, simplest definition. |
| **Tom Mitchell (1998)** | A program learns from **experience E** w.r.t. **task T** and **performance measure P**, if its performance at T (measured by P) **improves with E**. | The formal "T, P, E" definition. |
| **Ethem Alpaydın (2014)** | Programming computers to **optimize a performance criterion** using example data / past experience. | ML = optimization using data. |

#### 🧠 Worked example of Mitchell's T–P–E definition

Take a **spam filter**:

| Symbol | Meaning here |
|---|---|
| **T** (Task) | Classify an email as *spam* or *not spam* |
| **P** (Performance measure) | % of emails classified correctly (accuracy) |
| **E** (Experience) | Watching you mark emails as spam / not spam |

> If the filter's accuracy (P) on the task (T) **goes up** as it sees more of your marked emails (E) → by Mitchell's definition, **it is learning.** Always answer T–P–E questions by mapping each letter like this.

#### ✏️ Practice example (made up — solved)

Map Mitchell's T–P–E onto a **chess-playing program** that improves by playing games against itself:

| Symbol | Meaning here |
|---|---|
| **T** (Task) | Play a game of chess (choose good moves) |
| **P** (Performance measure) | % of games won (or its rating) |
| **E** (Experience) | The thousands of self-play games it records |

→ If the program's win-rate (P) at playing chess (T) climbs as it plays more self-games (E), it is **learning** by Mitchell's definition. (This is literally how AlphaZero was trained.)

---

## 2. Relationship between AI, ML and DL

### 🎯 Picture this

Think of **Russian nesting dolls** (matryoshka). The biggest doll contains a medium doll, which contains a small doll. AI is the biggest doll, ML sits inside it, and DL is the small doll inside ML. Everything inside is also part of everything outside it — but the outer doll has extra space the inner ones don't fill.

### 🔗 Why this matters

These three are **nested circles** — each is a subset of the bigger one. Mixing them up is a classic mistake; the nesting picture fixes it.

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

> 🔑 **The rule:** *All DL is ML, all ML is AI — but not all AI is ML.* (A rule-based chess engine is AI but not ML — it never learns; it just follows hand-written rules. It's the outer doll with nothing learning inside.)

---

## 3. Why do we require ML?

### 🎯 Picture this

You can instantly recognize your friend's face in a crowd — but try writing down the *exact rules* for "what my friend's face looks like" so a stranger could use them. Impossible: the rules are too many, too subtle, and you don't even consciously know them. Yet you could easily **show** someone 100 photos of your friend.

That gap — **easy to do, impossible to write as rules** — is exactly where ML is needed.

### 🔗 Why this matters

We need ML for tasks **easy for humans but very hard to hand-code**. You can't write "if-else" rules for "what does a cat look like" — but you *can* show 10,000 cat photos.

| Domain | Example tasks |
|---|---|
| **Vision** | Identify faces in a photo, detect objects in video/images |
| **Natural Language** | Translate Hindi → English, question answering, sentiment analysis |
| **Speech** | Recognize spoken words, speak sentences naturally |
| **Game playing** | Chess, Go, Dota |
| **Robotics** | Walking, jumping, displaying emotions |
| **Control** | Driving a car, navigating a maze |

> 💡 **Why hand-coding fails here:** the rules are either unknown to us, far too many, or constantly changing. ML lets the **data supply the rules**.

---

## 4. Traditional Programming vs ML

### 🎯 Picture this

A **recipe** vs. a **food critic learning to cook**.
- Traditional programming = following a **recipe**: you have the ingredients (data) and the written steps (program), and you produce a dish (output).
- Machine Learning = a person who has eaten **thousands of dishes and been told which were good** (data + answers), and from that *figures out the recipe themselves* (the program).

The recipe and the answers swap places. That's the whole contrast.

### 🔗 Why this matters

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

#### 🧠 Worked example (the "late payment" example)

**Goal:** Predict whether a customer will pay their bill late.

- **Traditional way:** A human writes rules like *"if income < X and missed > 2 payments → late."* Hard, brittle, you must already know the rule.
- **ML way:** You feed in:
  - **Input** = customer Demographics + past Bills
  - **Output** = "Paid late" or "Not late" (known historical answers)
  - The algorithm outputs a **"Late Payment Model"** (the recipe it figured out) that predicts for *new* customers.

> 🔑 **Quick test:** If the question gives you *answers* and asks for *rules* → it's ML. If it gives you *rules* and asks for *answers* → it's traditional programming.

#### ✏️ Practice example (made up — solved)

You want a system that decides if a photo contains a **traffic light**.

- **Traditional approach:** write rules — "look for a black box with one red, one yellow, one green circle…" Fails immediately (lights at night, sideways, partly hidden, different shapes worldwide).
- **ML approach:** collect 50,000 photos, each labelled "has traffic light" / "no traffic light" (Input + Output). The model produces its own internal rules. → Because we supplied **answers and want the rules**, this is **Machine Learning**.

---

## 5. The Recipe of ML (6 components)

### 🎯 Picture this

Baking a cake needs a fixed set of things: **ingredients**, a **goal** (a birthday cake vs. bread), a **recipe** you'll tweak, a way to **taste-test** how wrong it is, a **method to adjust** the recipe after tasting, and a **final judging** by guests. Skip any one and the cake fails.

Every ML system has the same six fixed ingredients.

### 🔗 Why this matters

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

### 5.1 Data — *"the ingredients"*

- Can be **structured** (tables/spreadsheets) or **unstructured** (images, text, audio).
- **Sources:** open dataset repositories, crowd-sourcing marketplaces, or make your own.
- **Splitting the data** (very important):
  - **Training data** → used to *teach* the model.
  - **Test data** → *unseen* data used afterward to *measure accuracy* honestly.

> 💡 **Why split?** Testing on the same data you trained on is like giving a student the exact exam questions to practise — they'd score 100% and you'd learn nothing about whether they truly understand. Test data must be *new*.

### 5.2 Task — *"what cake are we making?"*

The **type of prediction** being made.

| Task | What it does | Example |
|---|---|---|
| **Classification** | Assigns data to **categories** | Spam / not spam |
| **Clustering** | Groups data by **similarity** (no labels) | Customer segments |

### 5.3 Model — *"the recipe"*

> A **model** is just a **mathematical function** that maps **input → output**.

E.g. a model could be `output = a·x₁² + b·x₂³ + c·x₃²`. The letters `a, b, c` are **parameters** the machine must learn (the recipe's adjustable amounts).

Related ideas (covered deeply later): **bias–variance trade-off**, **regularization**, **overfitting** — all about making the recipe generalize to new dishes, not just memorize the ones tasted.

### 5.4 Loss Function — *"the taste-test: how bad is it?"*

> A **loss function** measures **how wrong** the model's predictions are. **Lower loss = better.** Learning = minimizing this number.

Common ones: **Mean Squared Error (MSE)**, **Cross-Entropy**.

#### 📐 Formula: Mean Squared Error (MSE)

$$\mathcal{L} = \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i)\right)^2$$

| Symbol | Meaning |
|---|---|
| $n$ | number of data points |
| $y_i$ | the **true** value for point $i$ |
| $\hat{f}(x_i)$ | the value the **model predicted** for point $i$ |
| $\left(y_i - \hat{f}(x_i)\right)$ | the **error** for one point (how far off the taste was) |
| squaring | makes all errors positive **and** punishes big errors more |

> *(The "mean" version divides by $n$; the slide shows the **sum** version — both are minimized the same way.)*

#### 🧠 Worked example of MSE

A model predicts house prices (in lakhs):

| House | True price $y_i$ | Predicted $\hat{f}(x_i)$ | Error $(y_i-\hat f)$ | Squared error |
|---|---|---|---|---|
| 1 | 50 | 48 | 2 | 4 |
| 2 | 60 | 65 | −5 | 25 |
| 3 | 55 | 54 | 1 | 1 |

**Step 1 — sum the squared errors:** $4 + 25 + 1 = 30$
**Step 2 — Sum-of-squares loss** $\mathcal{L} = \mathbf{30}$
**Step 3 — Mean squared error** $= 30 / 3 = \mathbf{10}$

> 👉 The learning algorithm's whole job is to adjust `a, b, c` until this number becomes as **small as possible**.

#### ✏️ Practice example (made up — solved)

A model predicts tomorrow's temperature (°C) for 4 days:

| Day | True $y_i$ | Predicted $\hat f(x_i)$ | Error | Squared error |
|---|---|---|---|---|
| 1 | 30 | 28 | 2 | 4 |
| 2 | 25 | 25 | 0 | 0 |
| 3 | 32 | 36 | −4 | 16 |
| 4 | 28 | 27 | 1 | 1 |

**Sum of squared errors:** $4 + 0 + 16 + 1 = \mathbf{21}$ → that's the loss $\mathcal{L}$.
**MSE:** $21 / 4 = \mathbf{5.25}$.
→ Notice day 3's big miss (−4) contributes 16 — most of the total — because squaring **punishes large errors much harder** than small ones. Driving this 21 down is the entire goal of training.

### 5.5 Learning Algorithm (Optimizer) — *"adjusting the recipe after tasting"*

> Job: **find the parameter values** (like `a, b, c`) that make the **loss minimum**.

$$\min_{a,b,c} \; \mathcal{L} = \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i)\right)^2$$

Reads: *"choose `a, b, c` that make the total error smallest."* Common optimizers: **Gradient Descent**, **AdaGrad**.

#### 🧠 Worked mini-example (the movie example)

Model: $\hat{f}(x) = a\,x_1^2 + b\,x_2^3 + c\,x_3^2$ predicting IMDB rating from (Budget, Box Office, Action-scene time). The optimizer tries many values and settles on:

| a | b | c | Loss | Min Loss |
|---|---|---|---|---|
| 20.0 | 20.0 | −10.0 | 3.74 | 3.74 |

→ It found that **a=20, b=20, c=−10** gives the **smallest loss (3.74)**. That trained set of numbers *is* the final model.

### 5.6 Evaluation — *"the guests judge the cake"*

After training, judge on **test data** using metrics: **Accuracy**, **Precision**, **Recall**, **F1-score** (detailed in Lecture 4).

> 🔑 **Summary chain:** Data feeds the Model; the Loss says how wrong it is; the Learning Algorithm tweaks parameters to shrink the loss; Evaluation gives the final report card.

---

## 6. Types of Learning

### 🎯 Picture this

Different ways a student can learn:
- With an **answer key** (supervised — every example comes with the right answer).
- **Sorting a pile** with no answer key, just grouping similar things (unsupervised).
- A **few answered examples** plus a big pile of unanswered ones (semi-supervised).
- Learning a video game by **trial and error**, guided only by the score (reinforcement).

The rest are variations on these themes.

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

### 6.1 Supervised Learning

🎯 **Story:** studying with a **fully solved answer key** — every practice question shows the correct answer, so you learn the input→answer mapping.

- Uses **labelled data** (every input comes with the correct answer).
- Learns a **mapping between inputs and the target variable**.

| Sub-type | Predicts | Example dataset |
|---|---|---|
| **Classification** | A **class label** (categories) | **MNIST**: handwritten digit image → digit 0–9 |
| **Regression** | A **numerical value** | **Boston housing**: neighbourhood info → house price ($) |

> 💡 **Memory hook:** **C**lassification → **C**ategories. **R**egression → **R**eal numbers.

**Common algorithms:** K-Nearest Neighbour, Linear Regression, Decision Tree, SVM.

```
   Labelled training data            New input
   (cat 🐱, dog 🐶, cat 🐱)  ──train──►  MODEL ──► "dog 🐶"
```

---

### 6.2 Unsupervised Learning

🎯 **Story:** dumped a box of mixed Lego on the floor with **no instructions** — you naturally start grouping by color/size. No one told you the "right" groups; you found structure yourself.

- Uses **unlabelled data** — only inputs, **no target**.
- Goal: **find hidden structure**.

| Sub-type | What it does | Example |
|---|---|---|
| **Clustering** | Find natural **groups** | Customer segmentation, image processing |
| **Density Estimation** | Summarize the **distribution** of data | Geographical & market data analysis |

**Common algorithms:** K-Means, Hierarchical clustering, Kernel Density Estimation, PCA.

```
   Unlabelled points         After clustering
     • •    • •                (Group A)  (Group B)
    •  •   • •      ──►        ◉ ◉ ◉      ▲ ▲ ▲
     • •     •                 ◉ ◉        ▲ ▲
```

> ⚠️ **Slide typo note:** the slide says *"Two main types of **supervised** learning problems"* under Unsupervised — it should read **unsupervised**.

---

### 6.3 Semi-Supervised Learning

🎯 **Story:** a teacher solves **3 problems** on the board, then hands you **300 unsolved** ones. The 3 worked examples guide how you tackle the giant unsolved pile.

- Uses **a few labelled + many unlabelled** examples.
- The few labels guide how to organize the huge unlabelled pile.
- **Semi-supervised clustering** uses the *known* cluster info to classify the *unlabelled* data.

**Example:** a **text document classifier** — learns from a small labelled set, then classifies a large unlabelled set.
**Common algorithms:** Co-training, Semi-supervised SVM.

> 💡 **Why it matters:** Labelling data is *expensive* (needs humans). Unlabelled data is *cheap and abundant*. Semi-supervised gets the best of both.

---

### 6.4 Reinforcement Learning (RL)

🎯 **Story:** learning a video game with **no manual**. You press buttons (actions), the game reacts, and the **score** tells you if that was good or bad. Over many tries you learn what works — purely from rewards.

- An **agent** acts in an **environment** and **learns from feedback** (rewards/punishments). No fixed dataset.

| Element | Meaning | Mario analogy |
|---|---|---|
| **Agent** | The program you train | Mario (AI-controlled) |
| **Environment** | The world it acts in | The game level |
| **Action** | A move that changes the state | Jump, move right |
| **Reward** | Evaluation of an action (+/−) | +points for coins, − for dying |

```
        ┌──────── ACTION ────────►┐
   ┌─────────┐                ┌──────────────┐
   │  AGENT  │                │ ENVIRONMENT  │
   └─────────┘                └──────────────┘
        ▲                            │
        └──── STATE, REWARD ─────────┘
```

**Common algorithms:** Q-Learning, Markov Decision Process (MDP).

> 🔑 **Key difference from supervised:** No "correct answer" is given. The agent discovers good behaviour through **trial and error guided by rewards**.

---

### 6.5 Bayesian Learning

🎯 **Story:** you think it *might* rain (a hunch — a "prior"). You see dark clouds (new evidence). You don't flip your belief 0→100; you **update it a bit** toward "rain." More evidence nudges it further. Beliefs evolve gradually with evidence.

- Based on **Bayes' Theorem**. Each new example **incrementally** raises/lowers the probability a hypothesis is correct (doesn't discard a hypothesis on one bad example — it's *flexible*).
- Combines **prior knowledge** with **observed data**.

#### 📐 Bayes' Theorem

$$P(H \mid D) = \frac{P(D \mid H)\,\cdot\,P(H)}{P(D)}$$

| Symbol | Name | Meaning |
|---|---|---|
| $P(H)$ | **Prior** | Belief in the hypothesis *before* seeing data |
| $P(D\mid H)$ | **Likelihood** | Probability of seeing this data *if* H were true |
| $P(H\mid D)$ | **Posterior** | Updated belief *after* seeing data (the answer) |
| $P(D)$ | **Evidence** | Total probability of the data (normalizer) |

#### 🧠 Worked example — the medical-test example

**Scenario:** A disease affects **1%** of people. The test is **99% accurate** for sick people, but has a **5% false-positive rate** for healthy people. **You test positive. What's the chance you're actually sick?**

- $H$ = "you have the disease". Prior $P(H) = 0.01$, so $P(\neg H)=0.99$.
- $D$ = "test is positive". Likelihoods: $P(D\mid H)=0.99$, $P(D\mid \neg H)=0.05$.

**Step 1 — total probability of a positive test:**
$$P(D) = P(D|H)P(H) + P(D|\neg H)P(\neg H) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594$$

**Step 2 — apply Bayes:**
$$P(H \mid D) = \frac{(0.99)(0.01)}{0.0594} = \frac{0.0099}{0.0594} \approx \mathbf{0.167}$$

> 😲 **Insight:** Even with a positive result on a "99% accurate" test, there's only a **~17% chance** you actually have the disease — because the disease is so rare (low prior). The prior matters enormously.

#### ✏️ Practice example (made up — solved)

**Spam filter.** 40% of all email is spam. The word "FREE" appears in **80%** of spam emails but only **10%** of normal emails. An email contains "FREE" — what's the chance it's spam?

- $H$ = "email is spam". Prior $P(H) = 0.4$, so $P(\neg H) = 0.6$.
- $D$ = "contains the word FREE". $P(D\mid H) = 0.8$, $P(D\mid \neg H) = 0.1$.

**Step 1 — total probability of seeing "FREE":**
$$P(D) = (0.8)(0.4) + (0.1)(0.6) = 0.32 + 0.06 = 0.38$$

**Step 2 — Bayes:**
$$P(H\mid D) = \frac{(0.8)(0.4)}{0.38} = \frac{0.32}{0.38} \approx \mathbf{0.84}$$
→ Seeing "FREE" jumps the spam belief from the prior **40% up to ~84%**. The evidence updated the belief — it didn't replace it outright (Bayesian flexibility in action).

**Common algorithms:** Naïve Bayes Classifier, Gibbs Algorithm.

---

### 6.6 Instance-Based Learning ("Lazy Learning")

🎯 **Story:** a student who **never studies in advance**. When a question appears, they flip through their notebook of past solved problems, find the most *similar* one, and copy its approach. All the "work" is postponed to question time.

- **Stores the training examples** instead of building a function up front.
- Generalization is **postponed until a new instance must be classified** → "lazy".
- New point → compared to **stored examples** to decide its label.

**Example:** **Recommender systems** — "users who liked X also liked Y".
**Common algorithms:** KNN, Locally Weighted Regression.

```
  Lazy (instance-based):  store all points → wait → compare on query
  Eager (e.g. linear reg): build one formula now → discard the data
```

> 💡 **Trade-off:** Lazy learners train instantly (just store) but predict slowly (compare against everything). Eager learners train slowly but predict fast.

---

### 6.7 Active Learning

🎯 **Story:** a smart student who, instead of doing *every* textbook problem, **picks only the confusing ones** and asks the teacher specifically about those. Far fewer questions asked, same understanding gained.

- An algorithm reaches **higher accuracy with fewer labelled examples** *if it gets to choose which data it learns from*.
- The **active learner asks queries**: picks the most informative unlabelled points and asks a **human (oracle)** to label just those.

```
        learn a model
   ┌────────────────────►  ML MODEL
   │                          │ selects the most
 LABELED set  ◄──────────┐    │ useful queries
   (ℒ)        labels them │    ▼
                       ORACLE ◄── UNLABELED pool (𝒰)
                    (human annotator)
```

- **Striking stat:** active learning needed only **31 labelled points** to reach the accuracy that passive (random) learning needed **174** points for → ~5.6× more label-efficient.

> 🔑 **Difference vs normal supervised:** Supervised is *passive* (takes whatever labels it's given). Active learning is *picky* (requests labels only for the most useful points).

---

### 6.8 Deep Learning (DL)

🎯 **Story:** recognizing a face in stages — first your eye detects **edges**, then edges combine into **shapes** (eye, nose), then shapes combine into a **whole face**. Each stage builds on the previous. A deep network stacks "stages" (layers) the same way.

- A **subfield of ML** using algorithms structured in **layers** → an **Artificial Neural Network (ANN)**.

```
  Input layer → Hidden layer 1 → Hidden layer 2 → ... → Output layer
    ●  ●  ●        ●  ●  ●           ●  ●            ●
    (each layer learns increasingly abstract features)
```

| Network | Best for |
|---|---|
| **Feedforward NN** | Basic prediction (data flows one way) |
| **Recurrent NN (RNN)** | Sequences — text, speech, time series |
| **Convolutional NN (CNN)** | Images and video |

> 💡 **Why "deep"?** "Depth" = many hidden layers. Each builds on the previous one's features (edges → shapes → objects), letting it solve very complex problems automatically.

---

## 7. Curse of Dimensionality

### 7.1 What is a "dimension"?

🎯 **Story:** finding a friend on a **1 km road** is easy (search a line). Finding them in a **1 km × 1 km field** is harder (search an area). In a **1 km³ multi-storey building**, much harder still (search a volume). Same-size space, but each added dimension makes the search blow up.

> **Dimension = number of features measured per data point.** Recording *age* and *income* = **2 dimensions**.

**As dimensions increase, problems pile up:**
- ⏱️ More **time** to compute
- 💾 More **memory**
- 🧩 More **complicated explanations** (regression with 100 parameters vs 2)
- 📊 **No simple visualization** (we can draw 2D/3D, not 10D)

> 🔑 **Definition:** the severe difficulties in **high-dimensional spaces** are called the **Curse of Dimensionality**.

### 7.2 Example 1 — the oil/water/gas pipeline data

- Measurements of an oil + water + gas mixture from pipelines.
- **3 categories:** homogeneous (red), annular (green), laminar (blue).
- The slide plots **100 points using only 2 of the features** ($x_6$, $x_7$) — but each point is really a **12-dimensional vector**.
- **Goal:** predict the class of a new point **×**.

### 7.3 The intuition (the grid/cell idea)

Where the new point **×** lands:
- Could be **Red** (many red points nearby)
- Could be **Green** (lots of green nearby)
- Unlikely **Blue** (few blue nearby)

> 🔑 **Core intuition:** *a point's class should be decided mostly by **nearby** training points.* (This single idea is the seed of **K-Nearest Neighbours**.)

**Simple solution — divide space into cells**, predict by which cell a point falls in (majority vote in that cell):

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

> ⚠️ **The curse strikes:** if each dimension is split into $k$ parts, total cells $= k^D$. The cells — and the data needed to fill them — **grow exponentially** with dimension $D$.

#### ✏️ Practice example (made up — solved)

Split each feature into $k = 10$ buckets. How many cells?

| Features $D$ | Cells $= 10^D$ |
|---|---|
| 1 | 10 |
| 2 | 100 |
| 4 | 10,000 |
| 8 | 100,000,000 |

→ Going from 2 features to 8 features explodes the cells from 100 to **100 million**. To have even *one* data point per cell you'd need 100 million examples. That's why high dimensions are "cursed" — the data demand grows out of control.

### 7.4 Example 2 — polynomial coefficients blow up

A general order-3 polynomial in $D$ inputs:

$$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{i=1}^{D} w_i x_i + \sum_{i=1}^{D}\sum_{j=1}^{D} w_{ij} x_i x_j + \sum_{i=1}^{D}\sum_{j=1}^{D}\sum_{k=1}^{D} w_{ijk} x_i x_j x_k$$

**Reading it:** term 1 = constant; term 2 = linear; term 3 = quadratic (pairs); term 4 = cubic (triples). Number of coefficients grows $\approx D^M$ for order $M$.

#### 🧠 Worked example — feel the explosion

Order $M = 3$, coefficients $\approx D^3$:

| $D$ (features) | Coefficients $\approx D^3$ |
|---|---|
| 2 | $2^3 = 8$ |
| 10 | $10^3 = 1{,}000$ |
| 100 | $100^3 = 1{,}000{,}000$ |

> 👉 Going from 10 to 100 features makes the model **1000× more complex**.

### 7.5 Remark + Solutions

- **More features ≠ better accuracy.** Adding features can make performance **worse**.
- Training examples needed grows **exponentially** with dimensionality ($\propto k^d$).

```
 performance
   │      ╭─╮
   │     ╱   ╲___          ← accuracy rises, peaks,
   │    ╱        ╲────___     then FALLS as you add
   │   ╱                  ─── too many features
   └──────────────────────────► dimensionality
       (this is the "Hughes phenomenon")
```

**✅ Solutions:**
- **Dimensionality reduction** (e.g. PCA — keep only the most important combined features)
- **Feature extraction / selection** (drop useless features, build informative ones)

> 🔑 **Takeaway:** Curse of dimensionality = *as dimensions grow, data needed grows exponentially, computation/memory explode, and accuracy can drop.* Fix it by **reducing dimensions**.

---

## 8. The Night-Before Recap

**ML** = learn patterns from data instead of being handed rules (teach a child "dog" by pointing, not by a rulebook). **Mitchell:** learns from **E** w.r.t. **T**, measured by **P**, if P on T improves with E.

**AI ⊃ ML ⊃ DL** — nesting dolls. Not all AI is ML (a rule-based engine is AI but never learns).

**Why ML:** for things easy to *do* but impossible to write as rules (recognize a face, a cat).

**Traditional vs ML:** recipe vs. learning the recipe from tasted dishes.
- Traditional: `Data + Program → Output`
- ML: `Data + Output → Program`

**ML Recipe (D-T-M-L-L-E):** Data, Task, Model, Loss function, Learning algorithm, Evaluation (the cake-baking ingredients).

**Loss (MSE):** $\mathcal{L}=\sum (y_i-\hat f(x_i))^2$ — square the errors (big errors punished harder), add them; learning minimizes it. (Worked: errors 2,−5,1 → loss 30, MSE 10. Practice: temps → loss 21, MSE 5.25.)

**Types of learning**

| Type | Data | Key word |
|---|---|---|
| Supervised | labelled | answer key; classification (categories) / regression (numbers) |
| Unsupervised | unlabelled | sort the Lego pile; clustering / density estimation |
| Semi-supervised | few labels + many unlabelled | 3 solved + 300 unsolved |
| Reinforcement | rewards | learn the game by score; agent/environment/action/reward |
| Bayesian | prior + data | update a hunch with evidence; Bayes' theorem |
| Instance-based | stored examples | lazy student flips notebook; KNN |
| Active | self-chosen | ask only about the confusing problems |
| Deep | big data | edges→shapes→face; layered neural nets |

**Bayes:** $P(H|D)=\dfrac{P(D|H)P(H)}{P(D)}$ — rare disease + positive test ⇒ still only ~17%. (Practice: "FREE" in email ⇒ spam belief 40% → 84%.)

**Curse of dimensionality:** searching a line vs. a building — cells/coefficients/data needed grow **exponentially** ($k^D$, $D^M$); more features can *hurt* accuracy. **Fix:** dimensionality reduction & feature extraction.

---

*✅ Study guide for Lecture 1–3 complete.*
