# Lecture 4: Logistic Regression, Decision Trees & Accuracy Metrics

> **Course:** BITS F464 — Machine Learning
> **This chapter has two parts:**
> - **Part A:** Perceptron, Logistic Regression, Gradient Descent
> - **Part B:** Decision Trees (ID3, Entropy, Information Gain) + Classification Accuracy Metrics
>
> **How this chapter is written:** every idea starts with a everyday-life story (no math), then connects that story to *why* we do the thing, and only then shows the formula — with every symbol explained in the language of the story. Every worked number from the original notes is kept. If you understand the story, the math is just the story written in symbols.

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
9. [The Night-Before Recap](#9-the-night-before-recap)

---

## 1. Perceptron

### 🎯 Picture this

Imagine you're deciding **whether to go outside and play**. You don't weigh every factor equally. "Is it raining?" matters a lot. "Is it a weekday?" matters a little. "Did I finish my homework?" matters somewhere in between. In your head you give each factor an **importance** and a **vote**, mentally add it all up, and if the total clears some personal **bar** — you go play.

A **perceptron** is exactly this decision-maker, written as a tiny machine. It's the **simplest artificial neuron** — the great-great-grandparent of every neural network in existence.

### 🔗 Why this matters

The whole reason this little machine is famous is one upgrade it made over an even older model. Watch the two side by side.

**The older model — McCulloch-Pitts neuron (no "importance", just a head-count):**

Think of a vote where **every person counts exactly the same**. You just count hands and check if enough went up.

$$y = 1 \;\text{ if } \sum_{i=0}^{n} x_i \geq \theta, \qquad y = 0 \;\text{ if } \sum_{i=0}^{n} x_i < \theta$$

- It just **adds up the inputs** and compares to a bar $\theta$ (theta). Inputs must be **Boolean** (0 or 1 — hand up or hand down).

**The perceptron — the upgrade: some voices matter more.** Now each input gets an **importance weight** $w_i$ (how much that factor's vote counts):

$$y = 1 \;\text{ if } \sum_{i=0}^{n} w_i x_i \geq \theta, \qquad y = 0 \;\text{ if } \sum_{i=0}^{n} w_i x_i < \theta$$

> 🔑 **The one difference that made it famous:** the perceptron added **numerical weights for inputs, plus a way to *learn* those weights from data**. And because of the weights, inputs are **no longer limited to just 0/1** — they can be any number (temperature in °C, age in years, etc.).

### 📐 The bias trick (a tidiness move)

Carrying that bar $\theta$ around separately is annoying. So we do a small algebra trick: move it to the other side.

$$\sum w_i x_i \geq \theta \quad\Longleftrightarrow\quad \sum w_i x_i - \theta \geq 0$$

Now invent a fake input that's **always 1** (call it $x_0 = 1$) and give it the weight $w_0 = -\theta$. The "$-\theta$" is now just another term in the sum, so everything collapses into one clean line:

$$y = 1 \;\text{ if } \sum_{i=0}^{n} w_i x_i \geq 0, \qquad y = 0 \;\text{ otherwise}$$

```
   x₀=1 ─w₀=-θ──┐
   x₁ ───w₁─────┤
   x₂ ───w₂─────┤──►(Σ wᵢxᵢ)──►[ ≥0? ]──► y (0 or 1)
   …             │
   xₙ ───wₙ─────┘
```

> 💡 **What the bias really is — in plain words:** $w_0$ is the machine's built-in **leaning** or **predisposition** *before it even looks at the inputs*. Like a strict teacher who needs strong reasons to say "yes" vs. a lenient one who says "yes" easily. Geometrically it lets the decision line sit anywhere, not just pass through the origin. Both the weights and the bias **depend on the data and must be learned**.

#### 🧠 Worked example

Weights $w = (w_0, w_1, w_2) = (-2, 1, 1)$ with the always-on $x_0=1$, and a real input $(x_1, x_2) = (3, 1)$:
$$\sum w_i x_i = (-2)(1) + (1)(3) + (1)(1) = -2 + 3 + 1 = 2$$
Since $2 \geq 0$ → the machine says **$y = 1$** (yes).

#### ✏️ Practice example (made up — solved)

**Should I go play?** Inputs: $x_1$ = "is it sunny?" (1=yes), $x_2$ = "did I finish homework?" (1=yes). Weights: $w_0 = -3$ (a cautious bias — needs a good reason to say yes), $w_1 = 2$ (sun matters), $w_2 = 2$ (homework matters). Always-on $x_0 = 1$.

*Case A — sunny but homework not done* $(x_1, x_2) = (1, 0)$:
$$\sum w_i x_i = (-3)(1) + (2)(1) + (2)(0) = -3 + 2 + 0 = -1$$
$-1 < 0$ → **$y = 0$** (don't play — one good reason isn't enough to clear the cautious bias).

*Case B — sunny and homework done* $(x_1, x_2) = (1, 1)$:
$$\sum w_i x_i = (-3)(1) + (2)(1) + (2)(1) = -3 + 2 + 2 = 1$$
$1 \geq 0$ → **$y = 1$** (play! two good reasons clear the bar). Notice how the negative bias $w_0$ is the "needs strong reasons" gatekeeper.

---

## 2. Perceptron Learning Algorithm

### 🎯 Picture this

You're learning to throw a basketball. You shoot — it sails too far left. So you **aim a bit more right** next time. Overshoot right? **Nudge back left.** You only ever correct yourself *when you miss*; when a shot goes in, you change nothing. Keep adjusting and eventually every shot goes in.

That's the entire perceptron learning algorithm. **"Only fix it when it's wrong, and nudge in the direction that would have made it right."**

### 🔗 Why this matters

A perceptron starts with **random weights** (random aim). It can't be useful until it tunes those weights so its yes/no answers match reality. The basketball "nudge after a miss" *is* the tuning rule.

**Setup:** $P$ = the inputs whose true answer is 1 (the "should-fire" points). $N$ = the inputs whose true answer is 0. Start with $\mathbf{w}$ random.

### 📐 The actual method

```
while not converged:
    pick a random point x from P ∪ N
    if x ∈ P  and  Σ wᵢxᵢ < 0  (should be +, but predicted −):
        w = w + x          ← nudge the boundary toward x
    if x ∈ N  and  Σ wᵢxᵢ ≥ 0  (should be −, but predicted +):
        w = w − x          ← nudge the boundary away from x
// done when EVERY input is classified correctly
```

> 💡 **Why "add x" / "subtract x"?** Adding the misclassified positive point's coordinates to $\mathbf{w}$ literally tilts the decision boundary *toward* counting that point as positive (aim more "right"). Subtracting does the opposite. Same as the basketball nudge.

**Absolute linear separability (the condition for this to ever finish):** two sets $P$ and $N$ are *absolutely linearly separable* if some weights $w_0, \dots, w_n$ exist where every point in $P$ gives $\sum w_i x_i > w_0$ and every point in $N$ gives $\sum w_i x_i < w_0$. In plain words: **a single straight line can cleanly separate the two groups**.

> ⚠️ **The famous catch:** the perceptron only **converges (finishes) if the data is linearly separable** — i.e., a single straight line *can* split the two classes. If no straight line works (the classic example is the **XOR** pattern), it never settles. This is the limitation Minsky & Papert pointed out in 1969, and it's why multi-layer networks were later invented.

#### 🧠 Worked example — one nudge step

Current $\mathbf{w} = (0, 1, -1)$ (bias included). A point $\mathbf{x} = (1, 2, 2)$ is in $P$ (true label 1).
- Score $= (0)(1) + (1)(2) + (-1)(2) = 0 + 2 - 2 = 0$. Since $0 \geq 0$ it predicts 1 → **correct, no change** (the shot went in).

Now take $\mathbf{x} = (1, 1, 3) \in P$ with $\mathbf{w} = (0, 1, -1)$:
- Score $= 0 + 1 - 3 = -2 < 0$ → predicted 0 but the truth is 1 → **missed**.
- Nudge: $\mathbf{w}_{new} = \mathbf{w} + \mathbf{x} = (0+1,\; 1+1,\; -1+3) = \mathbf{(1, 2, 2)}$.
- Re-check: $(1)(1)+(2)(1)+(2)(3) = 1+2+6 = 9 \geq 0$ → now correct ✓ (adjusted the aim, next shot goes in).

#### ✏️ Practice example (made up — solved)

Current $\mathbf{w} = (1, -1, 0)$ (bias, $w_1$, $w_2$). A point $\mathbf{x} = (1, 2, 1)$ is in $N$ (true label 0, "should NOT fire").

- Score $= (1)(1) + (-1)(2) + (0)(1) = 1 - 2 + 0 = -1$. Since $-1 < 0$ it predicts 0 → **correct, no nudge** (shot was fine).

Now a point $\mathbf{x} = (1, 0, 3) \in N$ (true label 0) with the same $\mathbf{w} = (1, -1, 0)$:
- Score $= (1)(1) + (-1)(0) + (0)(3) = 1 \geq 0$ → predicted 1 but the truth is 0 → **missed** (it's a negative point we wrongly fired on).
- Rule for a misclassified *negative* point: $\mathbf{w}_{new} = \mathbf{w} - \mathbf{x} = (1-1,\; -1-0,\; 0-3) = \mathbf{(0, -1, -3)}$.
- Re-check: $(0)(1) + (-1)(0) + (-3)(3) = 0 + 0 - 9 = -9 < 0$ → now predicts 0 → correct ✓ (we nudged the boundary *away* from this point).

#### 📜 Where it came from

| Year | Who | What |
|---|---|---|
| 1958 | **Frank Rosenblatt** | Proposed the classical perceptron |
| 1969 | **Minsky & Papert** | Refined it, and exposed the XOR limitation |

---

## 3. Logistic Regression — what & when

### 🎯 Picture this

A doctor doesn't usually say a flat *"you will have a heart attack"* or *"you won't."* They say something like **"based on your age, weight and blood pressure, there's about a 70% chance."** A number between 0 and 100%, not a hard yes/no — and *from* that percentage they then decide what to do.

**Logistic regression is that doctor.** It looks at the inputs and outputs a **probability of belonging to a class**, and only afterwards turns that probability into a yes/no.

### 🔗 Why this matters

> **Logistic regression** = a classification method for **binary (two-category) data** that gives you a **probability**, not just a label.

Despite the word "regression" in the name, its *job is classification*. The "regression" part is because **internally** it still fits a straight-line-style formula — it just squashes that formula's output into a 0-to-1 probability at the end (we'll see exactly how in §5).

**Real example:** a telecom company wants to predict which customers will **churn** (cancel next month). Others: probability of a heart attack from age/sex/BMI, or the chance a mortgage defaults.

### 📐 When to use it (the checklist)

- The thing you're predicting is **categorical / binary** (0/1, yes/no, churn/stay).
- You want the **probability**, not only the class (e.g. "73% likely to churn" lets you prioritize who to call).
- The data is roughly **linearly separable** (a line/plane can divide the classes).
- You want to **see which features matter** — the learned weights are readable (a big weight on "monthly bill" tells you bill size strongly drives churn).

**Two-class setup, in symbols:**

$$X \in \mathbb{R}^{m \times n} \quad (m \text{ examples}, n \text{ features}), \qquad y \in \{0, 1\}$$
$$\hat{y} = P(y=1 \mid x), \qquad P(y=0 \mid x) = 1 - P(y=1 \mid x)$$

Read $\hat{y} = P(y=1 \mid x)$ as: *"the model's predicted probability that this example is class 1, given its features."* And since the two probabilities must add to 1, the chance of class 0 is just whatever's left over.

The decision rule (from the slide diagram): predict class "yes" when $\theta_0 + \theta_1 x_1 + \theta_2 x_2 > 0$.

---

## 4. Linear vs Logistic Regression

### 🎯 Picture this

Imagine a **dimmer switch** vs. a **light switch**.

- A plain **linear** model is like asking the dimmer's *raw position number* and saying "if it reads above 0.5, the room is 'bright'." But the dimmer's number is unbounded — it could read −5 or +200. That's a meaningless quantity to call a probability, and the cutoff is a brittle, sudden cliff.
- **Logistic** regression replaces that brittle cliff with a **smooth ramp** that always stays between 0 and 1 — a genuine "how bright, as a fraction" reading you *can* treat as a probability.

### 🔗 Why this matters

This is *why* we don't just reuse linear regression for classification: its output isn't a probability and its hard step is fragile to outliers.

| | Linear Regression (used for classification) | Logistic Regression |
|---|---|---|
| Model | $\theta^T X = \theta_0 + \theta_1 x_1 + \dots$ | $\sigma(\theta^T X)$ |
| Output | A raw, unbounded number | A **probability** in [0, 1] |
| Decision | Hard **step**: $\hat y = 1$ if $\theta^T X \geq 0.5$, else 0 | Smooth S-curve → then threshold |
| Problem | Sharp jump, no probability, outlier-sensitive | Smooth, gives a real probability |

```
   STEP FUNCTION (linear)          SIGMOID (logistic)
 ŷ │      ┌──────                ŷ │          ╭───────
 1 │      │                      1 │       ╭──╯
   │      │   sharp cliff          │     ╭─╯  smooth S-ramp
 0 │──────┘                      0 │──╮─╯
   └──────┼────► θᵀX               └────┼────► θᵀX
        0.5                           0
```

> 🔑 **The sentence that captures it:** *Applying the sigmoid σ to θᵀx gives the **probability** of a point belonging to a class (logistic regression), instead of just the raw **value** of y (linear regression).*

---

## 5. The Sigmoid Function

### 🎯 Picture this

Think of how your **confidence** behaves as evidence piles up. With a little evidence you're unsure (~50/50). As good evidence accumulates, your confidence climbs toward 100% — but it can **never exceed 100%**, no matter how much more evidence arrives. It also never drops below 0%. It's a smooth **S-shaped** climb that flattens at both ends.

The **sigmoid** (a.k.a. logistic) function is the mathematical version of exactly that confidence curve. It takes *any* number — huge, tiny, negative — and gently squashes it into the range **(0, 1)**.

### 🔗 Why this matters

This squashing is the *entire* trick that turns a raw straight-line score into a usable probability (the "smooth ramp" from §4). The straight-line part $\theta^T X$ can be any number; the sigmoid is the funnel that forces it into a valid probability.

### 📐 The actual method

$$\sigma(\theta^T X) = \frac{1}{1 + e^{-\theta^T X}}$$

Decoding it like the confidence story:

```
 σ(θᵀX)
   1 │              ╭──────────  → σ → 1, P(y=1|x) goes UP
       │           ╭╯
  0.5 │·······╭───╯  ← σ(0) = 0.5 exactly (max uncertainty)
       │     ╭╯
   0 │─────╯              → σ → 0, P(y=1|x) goes DOWN
       └──┼──────────► θᵀX
          0
```

**Four facts to keep in your head (each is just a point on the confidence curve):**
- $\theta^T X$ **very big** (lots of evidence for "yes") → $\sigma \to 1$ → $P(y=1\mid x)$ goes **up**.
- $\theta^T X$ **very negative** (lots of evidence against) → $\sigma \to 0$ → $P(y=1\mid x)$ goes **down**.
- The output is **always in [0, 1]** → that's *why* we're allowed to read it as a probability.
- $\sigma(0) = 0.5$ exactly — the "I have no idea, it's a coin flip" middle, and the natural decision threshold.

#### 🧠 Worked example — compute the sigmoid

$\theta = (-1, 2)$ and input $X = (2, 5)$. First the straight-line score ($\theta^T X$ means the dot product):
$$\theta^T X = (-1)(2) + (2)(5) = -2 + 10 = 8$$
Now funnel that 8 through the sigmoid:
$$\sigma(8) = \frac{1}{1 + e^{-8}} = \frac{1}{1 + 0.000335} \approx \mathbf{0.9997}$$
→ $P(y=1\mid x) \approx 0.9997$ → the model is **very confidently** saying class 1.

*(The original slide uses the same setup $\theta=[-1,2]$, $X=[2,5]$ and rounds to $\hat y = \sigma(\dots) = 0.7$ — the point being identical: the sigmoid turns the score into a probability.)*

#### ✏️ Practice example (made up — solved)

A churn model has weights $\theta = (\theta_0, \theta_1, \theta_2) = (-4,\ 0.05,\ 1)$ where $x_1$ = monthly bill (₹), $x_2$ = number of complaints. A customer has bill $x_1 = 60$ and $x_2 = 1$ complaint. With the always-on bias input $x_0 = 1$:

**Step 1 — straight-line score** $\theta^T X$:
$$\theta^T X = (-4)(1) + (0.05)(60) + (1)(1) = -4 + 3 + 1 = 0$$

**Step 2 — funnel through the sigmoid:**
$$\sigma(0) = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = \mathbf{0.5}$$
→ $P(\text{churn}) = 0.5$ — the model is on the fence (exactly the "coin-flip" middle of the confidence curve).

**Now a second customer:** same weights, bill $x_1 = 100$, complaints $x_2 = 3$:
$$\theta^T X = (-4)(1) + (0.05)(100) + (1)(3) = -4 + 5 + 3 = 4$$
$$\sigma(4) = \frac{1}{1 + e^{-4}} = \frac{1}{1 + 0.0183} \approx \mathbf{0.98}$$
→ $P(\text{churn}) \approx 0.98$ — high bill + many complaints push the score up, so the model is very confident this customer will leave.

---

## 6. Training Process & Cost Function

### 🎯 Picture this

A student does practice problems. After each one they **check the answer key**, see how far off they were, and adjust their understanding. Do this over many problems and the gap between their answers and the truth shrinks. The "total amount they're still getting wrong" is a single score they're trying to drive down.

Training logistic regression is exactly this study loop, and the **cost function** is that single "how wrong am I overall" score.

### 🔗 Why this matters — the training loop

1. **Initialize** $\theta$ (start with a random guess at the weights).
2. **Predict** $\hat{y} = \sigma(\theta^T X)$ for a customer.
3. **Compare** $\hat{y}$ with the real answer $y$, record the **error** (check the answer key).
4. Do this for **all** customers (a full practice set).
5. **Adjust $\theta$** to make the total error smaller.
6. **Repeat** from step 2.

**Mini-example:** $\theta = [-1, 2]$, $X = [2, 5]$ → $\hat y = \sigma([-1,2]\cdot[2,5]) = 0.7$. True $y = 1$. **Error $= 1 - 0.7 = 0.3$.**

### 📐 The cost function (Cross-Entropy / Log Loss)

We need to write "how wrong" as a number. Here's the everyday logic first, *then* the formula.

**The idea:** punishment should depend on how **confidently wrong** you were. Saying "I'm 99% sure it's a cat" about a dog should hurt *far* more than saying "I'm 55% sure." A confident mistake is the worst kind.

For a **single example**, the cost is:

$$\text{Cost}(\hat{y}, y) = \begin{cases} -\log(\hat{y}) & \text{if } y = 1 \\ -\log(1 - \hat{y}) & \text{if } y = 0 \end{cases}$$

Why $-\log$? Because $-\log(\text{high probability you gave the right answer})$ is near 0 (barely punished), but $-\log(\text{tiny probability you gave the right answer})$ blows up toward infinity (savagely punished). That's the "confidently wrong = very bad" rule, in math.

Now combine all $m$ examples into one number:

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[\, y^i \log(\hat{y}^i) + (1 - y^i)\log(1 - \hat{y}^i) \,\Big]$$

> 💡 **The clever part — the $y$ and $(1-y)$ are on/off switches:**
> - If the true answer is $y = 1$: the $(1-y^i)$ becomes 0, so only the $\log(\hat y)$ term survives → cost $= -\log(\hat y)$. Predict $\hat y$ near 1 → cost ≈ 0 (good). Predict near 0 → cost → ∞ (you were confidently wrong, so it hurts a lot).
> - If the true answer is $y = 0$: the $y^i$ becomes 0, so only the $\log(1-\hat y)$ term survives. Same logic, mirrored.
>
> This is why cross-entropy is preferred over plain squared error for classification — it punishes *confident* mistakes much harder.

#### 🧠 Worked example — cross-entropy cost

3 examples. True labels $y = (1, 0, 1)$. Predicted $\hat y = (0.9, 0.2, 0.6)$.

Per-example cost (using the switch logic above):
- Example 1 ($y=1$, so use $-\log\hat y$): $-\log(0.9) = 0.105$
- Example 2 ($y=0$, so use $-\log(1-\hat y)$): $-\log(1-0.2) = -\log(0.8) = 0.223$
- Example 3 ($y=1$): $-\log(0.6) = 0.511$

$$J(\theta) = \frac{1}{3}(0.105 + 0.223 + 0.511) = \frac{0.839}{3} \approx \mathbf{0.28}$$
(Using natural log. Lower $J$ = the model is doing better.)

#### ✏️ Practice example (made up — solved)

Compare **two models** on the *same* 2 examples to see how the "confidently wrong" punishment works. True labels $y = (1, 0)$.

**Model A (good guesses):** $\hat y = (0.8, 0.1)$
- Example 1 ($y=1$): $-\log(0.8) = 0.223$
- Example 2 ($y=0$): $-\log(1-0.1) = -\log(0.9) = 0.105$
- $J_A = \frac{1}{2}(0.223 + 0.105) = \frac{0.328}{2} \approx \mathbf{0.16}$ (low cost — good)

**Model B (confidently wrong on example 1):** $\hat y = (0.05, 0.1)$
- Example 1 ($y=1$ but it said 0.05 — very wrong, very confident): $-\log(0.05) = 2.996$
- Example 2 ($y=0$): $-\log(0.9) = 0.105$
- $J_B = \frac{1}{2}(2.996 + 0.105) = \frac{3.101}{2} \approx \mathbf{1.55}$ (huge cost)

→ One confidently-wrong prediction (0.05 when the truth was 1) pushed Model B's cost from 0.16 to **1.55** — roughly **10× worse**. That single bad guess dominates the whole score, exactly the "confident mistakes are punished savagely" behaviour.

---

## 7. Gradient Descent

### 🎯 Picture this

You're standing somewhere on a **foggy hillside** and want to reach the **bottom of the valley**. You can't see far. But you *can* feel which way the ground slopes under your feet. Sensible plan: feel the steepest downhill direction, take a step that way, feel again, step again. Keep doing this and you walk yourself down to the lowest point.

That's **gradient descent**, exactly. The hillside is the cost $J(\theta)$, your position is the current weights, and "feeling the slope" is computing the gradient.

### 🔗 Why this matters

The cost function from §6 gives a single "how wrong" number for any setting of the weights $\theta$. Picture all possible weight settings spread out as a landscape, with height = cost. **Training = finding the lowest point in that landscape** (the weights that make the model least wrong). We can't see the whole landscape, so we walk downhill step by step.

```
   J(θ)
     \                         /
      \  ● start (high cost)  /
       \  ↓ step down        /
        \  ●                /
         \  ↓              /
          \_●_____________/   ← bottom of valley (lowest cost) = best θ
                θ
```

### 📐 The actual method

The **gradient** is just "which way is uphill, and how steep" — collected for every weight:

$$\nabla J = \left[\frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, \dots, \frac{\partial J}{\partial \theta_k}\right]$$

For logistic regression each piece works out to:
$$\frac{\partial J}{\partial \theta_1} = -\frac{1}{m}\sum_{i=1}^{m}(y^i - \hat{y}^i)\,x_1^i$$

**The update rule (the one to really know):**

$$\boxed{\;\theta_{new} = \theta_{prev} - \eta \,\nabla J\;}$$

| Symbol | In the hillside story |
|---|---|
| $\theta_{prev}$ | where you're standing now |
| $\eta$ (eta) | **learning rate** = how big a step you take |
| $\nabla J$ | the uphill direction (steepest *increase*) |
| **the minus sign** | so you step *downhill* — opposite of uphill |

> 💡 **Step-size (learning rate) trade-off:** tiny steps → you crawl down the hill, training takes forever. Huge steps → you leap right over the valley bottom and bounce around, never settling. You want a "just right" step — the Goldilocks size.

**Training loop with GD:**
1. Initialize weights randomly (drop yourself somewhere on the hill).
2. Feed the cost function the training set, compute the error.
3. Compute the **gradient** (feel the slope).
4. **Update**: $\theta_{new} = \theta_{prev} - \eta\nabla J$ (step downhill).
5. Repeat from step 2 until the cost is small enough (you've reached the valley).
6. Predict a new customer via $P(y=1\mid x) = \sigma(\theta^T X)$.

#### 🧠 Worked example — one downhill step

$\theta_{prev} = 0.8$, gradient $\nabla J = 2$, learning rate $\eta = 0.1$:
$$\theta_{new} = 0.8 - (0.1)(2) = 0.8 - 0.2 = \mathbf{0.6}$$
The gradient was positive (cost rises if you increase θ — i.e. uphill is toward bigger θ), so we moved θ **down** to 0.6. Repeat thousands of times → you arrive at the valley floor (the best θ).

#### ✏️ Practice example (made up — solved)

Let's take **3 steps** down the hill and watch the cost shrink. Suppose the cost is $J(\theta) = \theta^2$ (a simple bowl with its lowest point at $\theta = 0$). The slope of a bowl $\theta^2$ is $\nabla J = 2\theta$. Start at $\theta = 5$, learning rate $\eta = 0.1$.

**Step 1:** gradient $= 2(5) = 10$. $\theta_{new} = 5 - (0.1)(10) = 5 - 1 = \mathbf{4}$. Cost went $25 \to 16$.
**Step 2:** gradient $= 2(4) = 8$. $\theta_{new} = 4 - (0.1)(8) = 4 - 0.8 = \mathbf{3.2}$. Cost $16 \to 10.24$.
**Step 3:** gradient $= 2(3.2) = 6.4$. $\theta_{new} = 3.2 - (0.1)(6.4) = 3.2 - 0.64 = \mathbf{2.56}$. Cost $10.24 \to 6.55$.

→ θ marches 5 → 4 → 3.2 → 2.56 …, steadily walking down toward the valley bottom at θ = 0, and the cost keeps dropping. Notice the steps get **smaller** as the slope flattens near the bottom — that's gradient descent naturally easing off as it arrives.

**What if the step size is too big?** Same problem, but $\eta = 1.5$, start $\theta = 5$:
- Step 1: $\theta_{new} = 5 - 1.5(10) = 5 - 15 = \mathbf{-10}$ (cost jumped $25 \to 100$ — we leapt clean over the valley to the *other* side, higher than before). This is the "huge steps overshoot and never settle" failure.

---

## 8. Types of Gradient Descent

### 🎯 Picture this

You're polling people to decide which way to walk down the hill. You could (a) ask **everyone** before each step — accurate, but slow; (b) ask **one random person** and step immediately — fast but jittery; or (c) ask a **small handful** — a sensible compromise.

That's the only difference between the gradient-descent variants: **how much data you look at before each step.**

| Type | How it updates |
|---|---|
| **Batch GD** | Uses **all** training data for each step (accurate but slow — poll everyone) |
| **Stochastic GD (SGD)** | Uses **one** random example per step (fast, noisy — ask one person) |
| **Mini-batch GD** | Uses a **small batch** per step (best of both — the common default) |
| **Adagrad** | Adapts the step size per weight ⎫ |
| **RMSProp** | Root-mean-square propagation     ⎬ used in **Deep Learning** |
| **Adam** | Adaptive Moment Estimation       ⎭ |

> 🔑 The last three (Adagrad, RMSProp, Adam) are smarter optimizers used in deep learning — they automatically tune the step size as they go. **Adam** is the most popular default in practice.

---

## 9. The Night-Before Recap

**Perceptron** = a weighted vote machine. McCulloch-Pitts (no weights, Boolean inputs): $y=1$ if $\sum x_i \geq \theta$. Perceptron's upgrade: **learnable weights** $w_i$, non-Boolean inputs allowed. Bias trick: $x_0=1$, $w_0=-\theta$ → $y=1$ if $\sum w_i x_i \geq 0$.

**Perceptron learning** = basketball nudge: misclassified positive → $w = w + x$; misclassified negative → $w = w - x$. Only converges if the classes are linearly separable (one line can split them; fails on XOR).

**Logistic regression** = the doctor who gives a *probability* for a binary outcome. Use it when the target is binary, you want a probability, data is roughly separable, and you want readable feature impacts.

**Linear vs logistic** = brittle cliff vs. smooth ramp. Linear → raw unbounded number + hard step. Logistic → sigmoid → real probability.

**Sigmoid** = the confidence curve. $\sigma(\theta^T X) = \dfrac{1}{1+e^{-\theta^T X}}$, always in [0,1], $\sigma(0)=0.5$. Big score → 1; very negative → 0.

**Cost (cross-entropy / log loss)** = the "how confidently wrong were you" score:
$$J(\theta) = -\frac{1}{m}\sum \big[y\log\hat y + (1-y)\log(1-\hat y)\big]$$
The $y$ / $(1-y)$ act as switches; confident wrong answers are punished toward infinity. Lower = better.

**Gradient descent** = walking down a foggy hill: $\theta_{new} = \theta_{prev} - \eta\nabla J$. $\nabla J$ = uphill direction; minus = go downhill; $\eta$ = step size (too small = slow, too big = overshoot).

**Types of GD:** Batch (poll everyone), Stochastic (poll one), Mini-batch (poll a few — common); Adagrad/RMSProp/Adam (smart deep-learning optimizers).

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
15. [ID3 Algorithm — Full PlayTennis Worked Example](#15-id3-algorithm--full-playtennis-worked-example)
16. [Other Decision Tree Algorithms](#16-other-decision-tree-algorithms)
17. [Classification Accuracy Metrics](#17-classification-accuracy-metrics)
18. [The Night-Before Recap (Part B)](#18-the-night-before-recap-part-b)

---

## 10. What is a Decision Tree?

### 🎯 Picture this

Think about how you decide what to wear. You don't solve it all at once. You ask a **chain of small questions**: "Is it raining?" → if yes, grab a jacket; if no, "Is it cold?" → and so on. Each question is tiny and easy, and the chain of answers leads you to a decision.

A **decision tree** is that question-chain drawn out as a diagram. You start at the top, answer simple yes/no-ish questions, follow the matching branch, and end at a **leaf** that gives the answer.

### 🔗 Why this matters

> A **decision tree** is a predictive model built from a **branching series of simple tests**. Follow branches from the root down to a leaf = your prediction.

It's **inductive learning**: take specific examples (training data) and turn them into **general rules** (the tree). The reason we use a chain of tiny tests instead of one giant formula: **each small test is easy to make and easy to understand.** A hard decision becomes a sequence of trivial ones.

**The commute example — "How long will my commute take?":**

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

### 📐 Tree vocabulary

| Part | Meaning | In the PlayTennis example |
|---|---|---|
| **Root node** | The first/top question | Outlook |
| **Internal node** | A question on an attribute | Humidity, Wind |
| **Branch** | One possible answer to that question | Sunny, Rain, High |
| **Leaf node** | The final answer/class | Yes / No |

**The PlayTennis tree (we'll *build* this from scratch in §15):**
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

> 💡 **Reading the tree:** to classify "Sunny + High humidity" — start at Outlook → take the Sunny branch → arrive at Humidity → take the High branch → the leaf says **No** (don't play tennis). Just like the what-to-wear question chain.

---

## 11. Decision Tree Classification Task

### 🎯 Picture this

A teacher first **learns** what makes student answers right or wrong by grading a stack of *already-answered* papers (the answer key is known). Then they **apply** that learned judgement to grade a *new* batch of papers where the answers aren't yet marked.

A decision tree works the same way: learn the rules from labelled examples, then apply them to new, unlabelled data.

### 🔗 Why this matters — the workflow

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

- **Induction** = learning the tree from the training set (grading the known stack to learn the rules).
- **Deduction** = applying the learned tree to predict the test set's unknown classes (grading the new papers).

---

## 12. Decision Tree Learning Steps

### 🎯 Picture this

You're playing **20 Questions**. To win fast, your first question should be the one that **splits the possibilities most cleanly** — "Is it a living thing?" is a great opener; "Is it a platypus?" is a terrible one. After each answer you again pick the *most informative* next question for whatever's left.

Building a decision tree is automated 20 Questions: at every step, pick the question that best separates the remaining data.

### 🔗 Why this matters — the recipe

1. **Choose an attribute** from the dataset.
2. **Measure how well it splits the data** (how cleanly it separates the classes).
3. **Split the data** using the best attribute's values.
4. **Repeat** on each resulting subset.

> 🔑 **The one question this whole part answers:** *Which attribute is the "best" to split on?* The answer is the attribute with the **highest Information Gain** — and to compute that, we first need **Entropy**. Those are the next two sections.

---

## 13. Entropy

### 🎯 Picture this

Imagine a bag of marbles.
- A bag of **all red** marbles: if I ask "what's the next marble?" you know instantly — **zero uncertainty**, totally boring/pure.
- A bag that's **half red, half blue**: you genuinely can't guess — **maximum uncertainty**, totally mixed.

**Entropy is just a number for "how mixed/unpredictable is this bag."** All-one-color → entropy 0. Perfectly 50/50 → entropy 1 (its max for two classes). Anything in between → somewhere between.

### 🔗 Why this matters

A decision tree wants to make the data **less mixed** at every split (drive toward pure single-color bags = confident predictions). So we need a way to *measure* mixed-ness. That's entropy.

### 📐 The actual method

For events $\{A_1, \dots, A_n\}$ with probabilities $P(A_i)$:

$$H = -\sum_{i=1}^{n} P(A_i)\log_2 P(A_i)$$
(where $\sum P(A_i) = 1$ and $0 \le P(A_i) \le 1$)

The **self-information** of one event is $-\log_2 P(A_i)$ ("how surprising is this event"), and entropy is the **weighted average surprise** across all events.

**For two classes** (positives & negatives in a set $S$) — the version you'll actually use:

$$H(S) = -p_{+}\log_2(p_{+}) - p_{-}\log_2(p_{-})$$

where $p_+$ = fraction of positives, $p_-$ = fraction of negatives.

**Why "rarer = more informative" (the surprise idea):**

| Event probability | Information $-\log_2 P$ | Meaning |
|---|---|---|
| $P = 1$ (always happens) | $-\log_2(1) = 0$ | No surprise → **no info** |
| $P = 0.5$ (coin flip) | $-\log_2(0.5) = 1$ bit | Maximum uncertainty |
| $P = 0.001$ (rare) | $-\log_2(0.001) = 9.97$ | Big surprise → **lots of info** |

> 💡 **The intuition:** "The sun rose this morning" tells you nothing (it always happens, $P≈1$). "It snowed in Dubai today" is hugely informative ($P≈0$). Rare events carry more information — and entropy averages this surprise over the whole bag.

#### 🧠 Worked example — entropy of a set

A set with 9 positives and 5 negatives (total 14):
$$p_+ = \frac{9}{14}, \quad p_- = \frac{5}{14}$$
$$H(S) = -\frac{9}{14}\log_2\!\frac{9}{14} - \frac{5}{14}\log_2\!\frac{5}{14}$$
$$= -(0.643)(-0.637) - (0.357)(-1.486) = 0.410 + 0.530 = \mathbf{0.940}$$

> **Sanity checks against the marble bag:** an **all-positive** bag → $H = -1\log_2 1 - 0 = \mathbf{0}$ (no uncertainty, all one color). A **perfect 50-50** bag → $H = \mathbf{1}$ (max uncertainty). Our 0.940 is high → this set is quite mixed (slightly more positives).

#### ✏️ Practice example (made up — solved)

A bag of 10 emails: **8 spam, 2 not-spam**. How mixed is it?
$$p_+ = \frac{8}{10} = 0.8, \quad p_- = \frac{2}{10} = 0.2$$
$$H(S) = -0.8\log_2(0.8) - 0.2\log_2(0.2)$$
Compute each log (base 2): $\log_2(0.8) = -0.322$, $\log_2(0.2) = -2.322$.
$$H(S) = -(0.8)(-0.322) - (0.2)(-2.322) = 0.258 + 0.464 = \mathbf{0.722}$$
→ Lower than the 0.940 example, because 8-vs-2 is **less mixed** (more lopsided) than 9-vs-5. If it were 10 spam / 0 not-spam, $H$ would be exactly **0** (a pure, totally predictable bag).

---

## 14. Information Gain

### 🎯 Picture this

Back to 20 Questions. Before asking, you're very unsure (high entropy — mixed bag of possibilities). A **great** question chops that uncertainty down a lot; a **useless** question barely changes it. **Information Gain measures exactly how much a question cut the uncertainty.**

### 🔗 Why this matters

This is the scoring rule that picks the tree's questions: at each node, try every attribute, see which one reduces entropy the most, and use that one. **Biggest uncertainty-drop wins.**

### 📐 The actual method

$$\text{Gain}(S, A_i) = \underbrace{H(S)}_{\text{entropy before split}} - \underbrace{\sum_{v \in \text{Values}(A_i)} P(A_i = v)\,H(S_v)}_{\text{weighted entropy after split}}$$

| Part | Meaning (in the 20-Questions story) |
|---|---|
| $H(S)$ | uncertainty **before** asking (the mixed bag you start with) |
| $P(A_i = v)$ | fraction of data that gives answer $v$ to this question |
| $H(S_v)$ | uncertainty *within* the group that answered $v$ |
| The sum | the **average leftover uncertainty** after asking |

> 💡 **Plain English:** Gain = (uncertainty before) − (uncertainty after). Pick the attribute that **removes the most uncertainty** — that becomes the node. It's literally "which question teaches me the most."

#### ✏️ Practice example (made up — solved)

A set of **10 days**: 6 "Play", 4 "Don't play". Starting uncertainty:
$$H(S) = -\tfrac{6}{10}\log_2\tfrac{6}{10} - \tfrac{4}{10}\log_2\tfrac{4}{10} = -(0.6)(-0.737) - (0.4)(-1.322) = 0.442 + 0.529 = \mathbf{0.971}$$

Now we test the question **"Is it the weekend?"** which splits the 10 days into:
- **Weekend** (4 days): 4 Play, 0 Don't → pure → $H = \mathbf{0}$
- **Weekday** (6 days): 2 Play, 4 Don't → $H = -\tfrac{2}{6}\log_2\tfrac{2}{6} - \tfrac{4}{6}\log_2\tfrac{4}{6} = -(0.333)(-1.585) - (0.667)(-0.585) = 0.528 + 0.390 = 0.918$

Weighted leftover uncertainty after asking:
$$\tfrac{4}{10}(0) + \tfrac{6}{10}(0.918) = 0 + 0.551 = 0.551$$

**Information Gain:**
$$\text{Gain} = H(S) - 0.551 = 0.971 - 0.551 = \mathbf{0.420}$$
→ "Is it the weekend?" cut uncertainty from 0.971 down to 0.551, a gain of 0.420. If another attribute scored a *higher* gain, ID3 would pick that one for the node instead.

---

## 15. ID3 Algorithm — Full PlayTennis Worked Example

This is the worked example that ties Part B together. **ID3** (Iterative Dichotomiser 3) builds the tree by repeatedly playing the "pick the highest-information-gain question" game from §12–14.

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

### STEP 1 — Entropy of the whole set (how mixed is the starting bag?)

$$H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = \mathbf{0.940}$$

(That's the worked example from §13 — the bag is quite mixed.)

### STEP 2 — Information Gain for the attribute **Outlook**

Outlook has 3 possible answers. Split the 14 rows by them:

| Outlook | #Yes | #No | Total | Entropy of that subset |
|---|---|---|---|---|
| **Sunny** | 2 | 3 | 5 | $-\frac{2}{5}\log_2\frac25 - \frac{3}{5}\log_2\frac35 = \mathbf{0.971}$ |
| **Overcast** | 4 | 0 | 4 | $-\frac{4}{4}\log_2 1 - 0 = \mathbf{0.0}$ (pure! all Yes) |
| **Rain** | 3 | 2 | 5 | $-\frac{3}{5}\log_2\frac35 - \frac{2}{5}\log_2\frac25 = \mathbf{0.971}$ |

**Information Gain** = uncertainty before − weighted uncertainty after:
$$\text{Gain}(S,\text{Outlook}) = H(S) - \frac{5}{14}H(\text{Sunny}) - \frac{4}{14}H(\text{Overcast}) - \frac{5}{14}H(\text{Rain})$$
$$= 0.940 - \frac{5}{14}(0.971) - \frac{4}{14}(0) - \frac{5}{14}(0.971)$$
$$= 0.940 - 0.347 - 0 - 0.347 = \mathbf{0.246}$$

### STEP 3 — Do the same for the other attributes

(Identical process, looping over each attribute's values. Results:)

| Attribute | Information Gain |
|---|---|
| **Outlook** | **0.246** ← biggest uncertainty-drop |
| Temperature | 0.029 |
| Humidity | 0.029 |
| Wind | 0.048 |

### STEP 4 — Pick the root

> ✅ **Outlook has the highest gain (0.246) → Outlook becomes the ROOT node.** (It's the best opening question in our game of 20 Questions.)

### STEP 5 — Recurse on each branch

- **Outlook = Overcast** → entropy is **0** (all 4 days are "Yes") → a **pure leaf → answer: Yes.** Stop — no more questions needed for this branch (the bag is single-color).
- **Outlook = Sunny** (2 Yes, 3 No, $H = 0.971$) → still mixed, ask another question. Loop over the remaining attributes (Temp, Humidity, Wind):

  **Gain(S_Sunny, Temperature):** split the 5 Sunny rows by temperature:
  - Hot: 0 Yes, 2 No → H = 0
  - Mild: 1 Yes, 1 No → H = 1.0
  - Cool: 1 Yes, 0 No → H = 0
  $$\text{Gain} = 0.971 - \tfrac{2}{5}(0) - \tfrac{2}{5}(1.0) - \tfrac{1}{5}(0) = 0.971 - 0.4 = \mathbf{0.571}$$

  Similarly: **Gain(S_Sunny, Humidity) = 0.971** (the highest!), Gain(S_Sunny, Wind) = 0.020.

  → **Humidity** cuts the most uncertainty on the Sunny branch → it becomes the node under Sunny.

- **Outlook = Rain** → repeat the same game → **Wind** wins for that branch.

### STEP 6 — When does a branch stop growing?

A branch stops when **either**:
- Every attribute has **already been used** along that path, **OR**
- All training examples at that leaf have the **same class** (a pure, single-color bag).

> ⚠️ **One rule to remember:** an attribute used **higher** in the tree is **not reused** further down that same path.

### The final tree

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

> 🔑 **The recipe in order:** ① compute $H(S)$ → ② for each attribute, split & compute weighted child entropy → ③ Gain = $H(S)$ − weighted child entropy → ④ pick the max-gain attribute as the node → ⑤ recurse on each branch, dropping already-used attributes → ⑥ stop at pure leaves. (It's automated 20 Questions, every time.)

---

## 16. Other Decision Tree Algorithms

ID3 isn't the only tree builder. Same tree idea, different "best split" scoring or extra abilities:

| Algorithm | Note |
|---|---|
| **ID3** | Uses Information Gain (the method we just walked through) |
| **CART** | Classification **And** Regression Tree — can also predict *numbers*, not just classes; uses the **Gini index** instead of entropy |
| **C4.5** | An improved ID3 — uses *gain ratio*, and handles continuous data & missing values |

---

## 17. Classification Accuracy Metrics

Once any classifier is built, you ask: **how good is it really?** You check by comparing the **actual labels $y$** against the **predicted labels $\hat y$** on test data. Different metrics answer slightly different versions of "how good."

### 17.1 Jaccard Index

#### 🎯 Picture this

Two friends each list their favorite movies. How similar are their tastes? A fair measure: **(movies they both listed) ÷ (all distinct movies either listed)**. Total overlap → 1. No overlap → 0.

#### 📐 The method

$$J(y, \hat{y}) = \frac{|y \cap \hat{y}|}{|y \cup \hat{y}|} = \frac{|y \cap \hat{y}|}{|y| + |\hat{y}| - |y \cap \hat{y}|}$$

- $J = 1$ → perfect (the predicted set matches the actual set exactly). $J = 0$ → no overlap.

#### 🧠 Worked example

10 actual labels, 10 predicted, 8 of them match:
$$J = \frac{8}{10 + 10 - 8} = \frac{8}{12} = \mathbf{0.66}$$

#### ✏️ Practice example (made up — solved)

A spam filter is tested on **20 emails**. It predicts 20 labels; **15** of them match the true labels.
$$J = \frac{|y \cap \hat y|}{|y| + |\hat y| - |y \cap \hat y|} = \frac{15}{20 + 20 - 15} = \frac{15}{25} = \mathbf{0.6}$$
If it had gotten **all 20** right: $J = \frac{20}{20+20-20} = \frac{20}{20} = 1$ (perfect). If only **5** matched: $J = \frac{5}{35} \approx 0.14$ (poor overlap).

### 17.2 The Confusion Matrix (the foundation for the rest)

#### 🎯 Picture this

A **smoke alarm**. Four things can happen:
- Fire, alarm rings → **correctly caught** (True Positive).
- No fire, alarm silent → **correctly quiet** (True Negative).
- No fire, but alarm rings → **false alarm** (False Positive).
- Fire, but alarm stays silent → **missed the fire** (False Negative — the dangerous one).

The **confusion matrix** is just this 2×2 grid of "what happened vs. what the model said."

#### 📐 The method

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

| Term | Meaning | Smoke-alarm hook |
|---|---|---|
| **TP** (True Positive) | Correctly predicted **present** | Fire → alarm rang |
| **TN** (True Negative) | Correctly predicted **absent** | No fire → stayed quiet |
| **FP** (False Positive) | **Wrongly** said present | "False alarm" |
| **FN** (False Negative) | **Wrongly** said absent | "Missed the fire" |

> 💡 **Reading the names:** first word (True/False) = was the prediction **correct**? Second word (Positive/Negative) = **what did it predict**?

**Example confusion matrix used below:** TP=6, FN=9, FP=1, TN=24.

### 17.3 Precision, Recall, F1-score

#### 🎯 Picture this

A **fishing net**.
- **Precision** asks: of everything you pulled up in the net, what fraction was *actually fish* (not boots and seaweed)? High precision = few false alarms.
- **Recall** asks: of all the fish in the lake, what fraction did your net actually *catch*? High recall = few misses.
- There's a tension: a tiny careful net is very precise but misses lots of fish; a huge greedy net catches everything but hauls up junk too. **F1** is a single score that balances the two.

#### 📐 The method

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$

$$F1\text{-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Metric | The question it answers |
|---|---|
| **Precision** | Of all I flagged positive, how many really were? (avoids false alarms) |
| **Recall** | Of all the real positives, how many did I catch? (avoids misses) |
| **F1** | Harmonic mean — balances the two (especially useful when classes are imbalanced) |

#### 🧠 Worked example — using TP=6, FN=9, FP=1, TN=24

$$\text{Precision} = \frac{6}{6+1} = \frac{6}{7} = 0.86$$
$$\text{Recall} = \frac{6}{6+9} = \frac{6}{15} = 0.40$$
$$F1 = 2 \times \frac{0.86 \times 0.40}{0.86 + 0.40} = 2 \times \frac{0.344}{1.26} = 2 \times 0.273 = \mathbf{0.55}$$

(Matches the slide's "Churn=1: precision 0.86, recall 0.40, f1 0.55". Average accuracy across both classes was 0.72.)

#### ✏️ Practice example (made up — solved)

A disease screening test on 100 patients gives this confusion matrix: **TP = 40, FP = 10, FN = 5, TN = 45.** (40 sick people correctly flagged, 10 healthy people wrongly flagged, 5 sick people missed, 45 healthy people correctly cleared.)

$$\text{Precision} = \frac{TP}{TP+FP} = \frac{40}{40+10} = \frac{40}{50} = \mathbf{0.80}$$
*Of everyone the test flagged as sick, 80% really were sick (the net pulled up 80% fish, 20% junk).*

$$\text{Recall} = \frac{TP}{TP+FN} = \frac{40}{40+5} = \frac{40}{45} \approx \mathbf{0.89}$$
*Of all the truly sick people, the test caught 89% (it only missed 5 of 45 — good, important for a disease test).*

$$F1 = 2 \times \frac{0.80 \times 0.89}{0.80 + 0.89} = 2 \times \frac{0.712}{1.69} = 2 \times 0.421 \approx \mathbf{0.84}$$
*The single balanced score is 0.84 — high recall (few misses) and decent precision combine into a strong F1.*

> 💡 **Why not just use accuracy?** If 95% of emails are "not spam", a lazy model that *always* says "not spam" scores 95% accuracy — yet it never catches a single spam (recall for spam = 0). The fishing-net view (precision/recall/F1) exposes that uselessness; raw accuracy hides it.

### 17.4 Log Loss

#### 🎯 Picture this

This is the **same "how confidently wrong were you" punishment** from Part A §6 — reused here as a *grade* for the finished model instead of a training signal. Confidently right → barely penalized. Confidently wrong → heavily penalized.

#### 📐 The method

$$\text{LogLoss} = -\frac{1}{n}\sum\Big[\, y \times \log(\hat{y}) + (1 - y)\times\log(1 - \hat{y}) \,\Big]$$

- It's literally the **cross-entropy formula from §6**, used here as an evaluation metric.
- **Lower LogLoss = better** (note: opposite direction from Jaccard/F1, where higher = better).
- It punishes confident wrong probabilities. (A true 1 predicted as 0.13 → LogLoss 2.04, very high; predicted as 0.91 → LogLoss 0.11, low.)

#### 🧠 Worked example

True $y=1$, predicted $\hat y = 0.9$: contribution $= -[1\cdot\log(0.9) + 0\cdot\log(0.1)] = -\log(0.9) = 0.105$.
True $y=0$, predicted $\hat y = 0.2$: contribution $= -[0 + 1\cdot\log(0.8)] = -\log(0.8) = 0.223$.
Average over the whole dataset → final LogLoss (the slide's example totals to 0.60).

> 🔑 **Direction summary:** Jaccard, Precision, Recall, F1 → **higher is better**. LogLoss → **lower is better**.

---

## 18. The Night-Before Recap (Part B)

**Decision tree** = a chain of small questions (like deciding what to wear). Root → internal nodes → leaves. Inductive learning (examples → general rules).

**Tree learning** = automated 20 Questions: choose attribute → measure how well it splits → split → repeat. The best attribute = highest Information Gain.

**Entropy** = "how mixed is the marble bag": $H(S) = -p_+\log_2 p_+ - p_-\log_2 p_-$.
- All one color → H=0. Perfect 50-50 → H=1. Rarer event = more surprise/info ($-\log_2 P$).
- 9 pos / 5 neg of 14 → H = **0.940**.

**Information Gain** = how much a question cut the uncertainty = $H(S) - \sum P(A=v)H(S_v)$. Pick **max gain**.

**ID3 PlayTennis:** H(S)=0.940 → Gain(Outlook)=**0.246** (max), Temp=0.029, Humidity=0.029, Wind=0.048 → **Outlook = root**. Overcast branch is a pure leaf (Yes). Recurse: Sunny→Humidity, Rain→Wind. Drop used attributes; stop at pure leaves.

**Other algorithms:** CART (classification + regression, Gini index), C4.5 (improved ID3, gain ratio, handles continuous/missing).

**Confusion matrix** = the smoke-alarm grid: TP / TN (correct), FP (false alarm) / FN (missed it).

| Metric | Formula | Direction |
|---|---|---|
| Jaccard | $\frac{|y\cap\hat y|}{|y|+|\hat y|-|y\cap\hat y|}$ | higher better |
| Precision | $\frac{TP}{TP+FP}$ | higher better |
| Recall | $\frac{TP}{TP+FN}$ | higher better |
| F1 | $2\frac{P\cdot R}{P+R}$ | higher better |
| LogLoss | $-\frac1n\sum[y\log\hat y+(1-y)\log(1-\hat y)]$ | **lower** better |

Worked: Jaccard 8/(10+10−8)=0.66. TP6/FN9/FP1/TN24 → P=0.86, R=0.40, F1=0.55.

---

*✅ Lecture 4 (Logistic Regression + Decision Trees + Accuracy Metrics) — both parts complete.*
