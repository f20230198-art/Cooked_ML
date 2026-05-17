# Lecture 5: Artificial Neural Networks (ANN)

> **Course:** BITS F464 — Machine Learning
> **Covers:** Biological inspiration, McCulloch-Pitts neuron (+ Boolean gates, geometry), Perceptron & its learning, Sigmoid neuron, Multilayer Perceptron (MLP), Feed-Forward Neural Network, decision boundaries by depth, activation functions, network training, and **Backpropagation** (the chain rule).
>
> **How this chapter is written:** every idea opens with an everyday story, connects it to *why*, then the method/formula with symbols explained in the story's language. All original worked examples kept; important formulas also get an **extra made-up example, fully solved**.

> 📌 *Note:* some of this (Perceptron, sigmoid, gradient descent) repeats Lecture 4. Those parts are brief here with cross-references; see [04_LogisticReg_DecisionTree_Metrics.md](04_LogisticReg_DecisionTree_Metrics.md) Part A for the full treatment. The **new** material is MP-neuron Boolean gates, MLP, FFNN equations, and backprop.

---

## Table of Contents

1. [Biological Inspiration](#1-biological-inspiration)
2. [McCulloch-Pitts (MP) Neuron](#2-mcculloch-pitts-mp-neuron)
3. [Boolean Functions using MP](#3-boolean-functions-using-mp)
4. [Geometric Interpretation & Limitation](#4-geometric-interpretation--limitation-of-mp)
5. [Perceptron (recap + worked example)](#5-perceptron-recap--worked-example)
6. [Multilayer Perceptron (MLP)](#6-multilayer-perceptron-mlp)
7. [Perceptron vs Sigmoid Neuron](#7-perceptron-vs-sigmoid-neuron)
8. [Feed-Forward Neural Network (FFNN)](#8-feed-forward-neural-network-ffnn)
9. [Decision Boundary by Network Depth](#9-decision-boundary-by-network-depth)
10. [Activation Functions](#10-activation-functions)
11. [Training a Neural Network](#11-training-a-neural-network)
12. [Backpropagation](#12-backpropagation)
13. [The Night-Before Recap](#13-the-night-before-recap)

---

## 1. Biological Inspiration

### 🎯 Picture this

A single brain cell (neuron) is like a tiny **decision desk**. Messages arrive on many incoming wires (dendrites), the desk weighs them and decides whether the message is strong enough to **pass it on**, and if so it fires a signal down one outgoing wire (axon) to the next desk. An artificial neuron is a stripped-down copy of this desk.

### 🔗 Why this matters

| Biological part | Job | ANN equivalent |
|---|---|---|
| **Dendrite** | Receives inputs | Inputs $x_1, x_2, \dots$ |
| **Synapse** | Connection strength | Weights $w_1, w_2, \dots$ |
| **Soma** | Processes the info | The aggregation + activation |
| **Axon** | Transmits the output | Output $y$ |

```
   Biological:  dendrites → soma → axon → (synapse) → next neuron
   Artificial:  inputs xᵢ → Σ(wᵢxᵢ) → activation σ → output y
```

> 💡 **Memory hook:** **D**endrite = **D**ata in. **A**xon = **A**nswer out. Synapse = the *strength* of a connection = the weight.

---

## 2. McCulloch-Pitts (MP) Neuron

### 🎯 Picture this

A very strict committee that **just counts raised hands** — no member's vote counts more than another's. If at least $\theta$ hands go up, the motion passes (output 1); otherwise it fails (output 0). And there's one veto rule: if *any* member raises an "inhibitory" hand, the motion fails outright no matter what.

### 🔗 Why this matters

The **first computational model of a neuron (1943)** — deliberately crude.

$$g(x_1, \dots, x_n) = \sum_{i=1}^{n} x_i, \qquad y = \begin{cases} 1 & \text{if } g(\mathbf{x}) \geq \theta \\ 0 & \text{if } g(\mathbf{x}) < \theta \end{cases}$$

- $g$ aggregates (just sums). $f$ decides via threshold $\theta$.
- Inputs **excitatory** (encourage firing) or **inhibitory** (veto). **$y=0$ if ANY input is inhibitory.**
- Inputs/output **Boolean** (0/1). **No weights** (that's the perceptron's later upgrade).

```
   x₁ ─┐
   x₂ ─┼──►( g = Σxᵢ )──►[ ≥ θ ? ]──► y ∈ {0,1}
   xₙ ─┘
```

#### 🧠 Worked example

3 inputs $(1,1,0)$, $\theta = 2$, none inhibitory: $g = 1+1+0 = 2$, $2 \geq 2 \Rightarrow y = 1$.
If $\theta = 3$: $g = 2 < 3 \Rightarrow y = 0$.

#### ✏️ Practice example (made up — solved)

4 inputs $(1, 0, 1, 1)$, threshold $\theta = 3$, none inhibitory:
$$g = 1+0+1+1 = 3, \qquad 3 \geq 3 \Rightarrow y = \mathbf{1}$$
Now the **veto rule**: same inputs, but suppose input 2 is **inhibitory** (even though it's 0, the rule is about the input *being* inhibitory and active). If any inhibitory input fires → $y = \mathbf{0}$ regardless of the sum. This is the hard "any veto kills it" behaviour.

---

## 3. Boolean Functions using MP

### 🎯 Picture this

By moving the "how many hands needed" bar, the same committee becomes different logic gates:
- Set the bar to "**everyone** must agree" → that's an **AND** gate.
- Set the bar to "**at least one** is enough" → that's an **OR** gate.

### 🔗 Why this matters

(Assume 3 inputs, each 0/1.)

| Gate | Threshold $\theta$ | Fires (y=1) when |
|---|---|---|
| **AND** | **3** | ALL three inputs are 1 (sum = 3 ≥ 3) |
| **OR** | **1** | AT LEAST one input is 1 (sum ≥ 1) |

#### 🧠 Worked example — verify AND ($\theta = 3$)

| $x_1$ | $x_2$ | $x_3$ | sum | ≥3? | $y$ |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 3 | yes | **1** ✓ |
| 1 | 1 | 0 | 2 | no | **0** ✓ |
| 0 | 0 | 0 | 0 | no | **0** ✓ |

#### 🧠 Worked example — verify OR ($\theta = 1$)

| $x_1$ | $x_2$ | $x_3$ | sum | ≥1? | $y$ |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | no | **0** ✓ |
| 1 | 0 | 0 | 1 | yes | **1** ✓ |
| 0 | 1 | 1 | 2 | yes | **1** ✓ |

#### ✏️ Practice example (made up — solved)

Design a **2-input AND** gate, then test it. For an $n$-input AND, set $\theta = n$, so here $\theta = 2$.

| $x_1$ | $x_2$ | sum | ≥2? | $y$ | AND correct? |
|---|---|---|---|---|---|
| 0 | 0 | 0 | no | 0 | 0 AND 0 = 0 ✓ |
| 1 | 0 | 1 | no | 0 | 1 AND 0 = 0 ✓ |
| 0 | 1 | 1 | no | 0 | 0 AND 1 = 0 ✓ |
| 1 | 1 | 2 | yes | **1** | 1 AND 1 = 1 ✓ |

→ Only fires when *both* are 1 = exactly AND. (For a 2-input OR you'd instead set $\theta = 1$.)

> 🔑 For an $n$-input AND, $\theta = n$. For OR, $\theta = 1$.

---

## 4. Geometric Interpretation & Limitation of MP

### 🎯 Picture this

Plot the input combinations as dots on graph paper, coloring "output 1" dots black and "output 0" dots white. The condition $\sum x_i = \theta$ is a **straight line**. The neuron works only if you can draw that **one straight line with all black dots on one side and all white dots on the other**. Some patterns make that impossible — no matter how you angle the line.

### 🔗 Why this matters

```
   OR (θ=1):  x₁+x₂ ≥ 1            AND (θ=2):  x₁+x₂ ≥ 2
   1 ●────────● (1,1)→1            1 ○        ● (1,1)→1
     │╲                              │      ╲
   0 ○──╲─────● (1,0)→1            0 ○────────○
     (0,0)→0    x₁                  (0,0)→0  (1,0)→0
```

> 🔑 **LIMITATION:** a **single MP neuron can only represent Boolean functions that are LINEARLY SEPARABLE** (one straight line splits the 1s from the 0s).

> ⚠️ AND, OR are linearly separable (one line works). **XOR is NOT** — its black/white dots are diagonally opposite, so no single line can separate them. This famous limitation motivated **multilayer** networks (§6).

---

## 5. Perceptron (recap + worked example)

> 📌 Full treatment + the "weighted vote" analogy in [Lecture 4 Part A](04_LogisticReg_DecisionTree_Metrics.md). Brief recap:

🎯 **Story:** the MP committee where now **some members' votes count more** (weights) — and the committee can *learn* those weights from mistakes (the basketball-aim nudge from Lecture 4).

The **Perceptron** (Rosenblatt 1958; refined Minsky & Papert 1969) adds **learnable weights** $w_i$; inputs **no longer Boolean**.

$$y = 1 \text{ if } \sum_{i=0}^{n} w_i x_i \geq 0, \quad \text{else } 0 \qquad (x_0=1,\; w_0=-\theta)$$

**Learning:** misclassified positive → $\mathbf{w}=\mathbf{w}+\mathbf{x}$; misclassified negative → $\mathbf{w}=\mathbf{w}-\mathbf{x}$. Converges only if **linearly separable**.

#### 🧠 Worked example — single artificial neuron

Weights $w_1=2, w_2=-4, w_3=1$, activation $\varphi(v)=1$ if $v\geq0$ else 0, $v = w_1x_1+w_2x_2+w_3x_3$.

- **P1:** $(1,0,0)$ → $v = 2(1)-4(0)+1(0) = 2 > 0$ → $y = \mathbf{1}$
- **P2:** $(0,1,1)$ → $v = 2(0)-4(1)+1(1) = -3 < 0$ → $y = \mathbf{0}$

#### ✏️ Practice example (made up — solved)

Weights $w_1 = -1, w_2 = 3, w_3 = 2$, same step activation, $v = w_1x_1+w_2x_2+w_3x_3$.

- **Input $(2, 1, 0)$:** $v = (-1)(2) + (3)(1) + (2)(0) = -2 + 3 + 0 = 1$. $1 \geq 0$ → $y = \mathbf{1}$
- **Input $(3, 0, 1)$:** $v = (-1)(3) + (3)(0) + (2)(1) = -3 + 0 + 2 = -1$. $-1 < 0$ → $y = \mathbf{0}$

> ✅ **Recipe:** ① compute $v=\sum w_i x_i$ → ② step: $v\geq0\to1$, else 0.

---

## 6. Multilayer Perceptron (MLP)

### 🎯 Picture this

One person can't solve a hard problem alone, but a **team in layers** can: front-line workers each handle a small sub-question, a manager combines their answers. Stacking neurons into layers lets the network solve problems a single neuron can't (like XOR).

### 🔗 Why this matters

```
        y          ← Output layer: 1 neuron
       ╱│╲╲
   h₁ h₂ h₃ h₄     ← Hidden layer: 4 perceptrons
    ╲╲╳╳╱╱
   x₁    x₂         ← Input layer: 2 inputs
```

| Layer | Contents |
|---|---|
| **Input** | $x_1, x_2$ (raw features) |
| **Hidden** | 4 perceptrons → $h_1..h_4$ |
| **Output** | 1 neuron → final $y$ |

> 🔑 **The rule:** *Any Boolean function of $n$ inputs can be represented by a network with **1 hidden layer of $2^n$ perceptrons** + 1 output perceptron.*

#### 🧠 Worked example

$n = 3$ inputs → hidden layer needs $2^n = 2^3 = \mathbf{8}$ perceptrons + 1 output.

> 💡 **Why $2^n$?** A Boolean function of $n$ inputs has $2^n$ possible input combinations; worst case you need one hidden neuron to "detect" each, then OR them at the output.

#### ✏️ Practice example (made up — solved)

You need to represent an arbitrary Boolean function of **$n = 4$** inputs. By the rule:
$$\text{hidden perceptrons} = 2^n = 2^4 = \mathbf{16}, \qquad \text{plus } 1 \text{ output perceptron}$$
→ 16 hidden + 1 output. (Note how fast this grows: $n=3\to8$, $n=4\to16$, $n=5\to32$ — doubling each time, since each new input doubles the number of input combinations to cover.)

---

## 7. Perceptron vs Sigmoid Neuron

### 🎯 Picture this

A **light switch** vs. a **dimmer knob**. The perceptron's step function is a switch — it slams from OFF to ON with no in-between, and there's no notion of "how close to flipping" it is. The sigmoid is a dimmer — a smooth ramp. Why does smoothness matter? Because to *train* by "which tiny tweak reduces error," you need to measure slope — and a switch has no usable slope (it's a vertical cliff), while a dimmer's ramp has slope everywhere.

### 🔗 Why this matters

| | **Perceptron (step)** | **Sigmoid neuron** |
|---|---|---|
| Output | $y=1$ if $\sum w_ix_i \geq 0$ else 0 | $y = \dfrac{1}{1+e^{-\sum w_ix_i}}$ |
| Shape | Hard step | Smooth S-curve |
| Properties | Not smooth, **NOT differentiable** | Smooth, **DIFFERENTIABLE** |

> 🔑 **Why differentiability matters:** training uses **gradient descent**, which needs **derivatives** (slopes). The step has no usable slope → can't compute gradients → can't train deep nets. The sigmoid IS differentiable → enables **backpropagation**. *This is why neural nets use smooth activations.*

---

## 8. Feed-Forward Neural Network (FFNN)

### 🎯 Picture this

An **assembly line**. Raw material enters at one end. Each station does **two steps**: combine parts (a weighted sum), then apply a finishing process (squash through an activation). The output of one station is the input to the next. Material only flows **forward** — never loops back.

### 🔗 Why this matters

A special MLP where data flows strictly forward (input → hidden → output, no loops).

| Symbol | Meaning |
|---|---|
| Input layer | N-dim vector; the **0th layer** |
| Hidden layers | $L-1$ hidden layers |
| Output layer | $k$ neurons; the **Lth layer** |
| $a_i$ | **pre-activation** at layer $i$ (the linear part) |
| $h_i$ | **activation** at layer $i$ (after squashing) |
| $W_i, b_i$ | weight & bias between layers $i{-}1$ and $i$ |

### 📐 The two key equations

**Pre-activation** (combine parts — weighted sum + bias):
$$a_i(x) = b_i + W_i\, h_{i-1}(x)$$

**Activation** (finishing process — squash through $g$):
$$h_i(x) = g\big(a_i(x)\big)$$

Final output uses an output activation $O$: $f(x) = h_L(x) = O\big(a_L(x)\big)$.

```
  x ─► [a₁ = b₁ + W₁x] ─► [h₁ = g(a₁)] ─► [a₂ = b₂ + W₂h₁] ─► … ─► ŷ
       └─ pre-activation ┘ └─ activation ┘   (repeat per station)
```

> 💡 **Two-step rhythm per layer:** ① linear (weighted sum + bias) → $a$. ② non-linear (squash) → $h$. That $h$ feeds the next layer. Repeat.

#### 🧠 Worked example — one layer

Inputs $h_0 = (2, 3)$. $W_1 = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix}$, $b_1 = (1, -1)$, sigmoid activation.

Pre-activation:
- Neuron 1: $a_{11} = 1 + (1\cdot 2 + 0\cdot 3) = 1 + 2 = 3$
- Neuron 2: $a_{12} = -1 + (-1\cdot 2 + 2\cdot 3) = -1 + 4 = 3$

Activation:
$$h_{11} = \sigma(3) = \frac{1}{1+e^{-3}} \approx 0.95, \qquad h_{12} = \sigma(3) \approx 0.95$$
→ $h_1 = (0.95, 0.95)$ feeds the next layer.

#### ✏️ Practice example (made up — solved)

Inputs $h_0 = (1, 2)$. $W_1 = \begin{bmatrix} 2 & 1 \\ 0 & -1 \end{bmatrix}$, $b_1 = (-1, 3)$, sigmoid activation.

**Pre-activation $a_1 = b_1 + W_1 h_0$:**
- Neuron 1: $a_{11} = -1 + (2\cdot 1 + 1\cdot 2) = -1 + (2 + 2) = -1 + 4 = 3$
- Neuron 2: $a_{12} = 3 + (0\cdot 1 + (-1)\cdot 2) = 3 + (0 - 2) = 3 - 2 = 1$

**Activation $h_1 = \sigma(a_1)$:**
$$h_{11} = \sigma(3) = \frac{1}{1+e^{-3}} \approx \mathbf{0.95}, \qquad h_{12} = \sigma(1) = \frac{1}{1+e^{-1}} = \frac{1}{1.368} \approx \mathbf{0.73}$$
→ $h_1 = (0.95, 0.73)$ becomes the input to the next station. Same two-step rhythm, repeated until the output.

### Supervised learning in FFNN

- **Data:** $\{x_i, y_i\}_{i=1}^{N}$
- **Model:** $\hat{y}_i = O(W^3 g(W^2 g(W^1 x + b_1) + b_2) + b_3)$ (nested layers)
- **Algorithm:** Gradient Descent **with Backpropagation**
- **Loss:** e.g. $\min \dfrac{1}{N}\sum (\hat y_i - y_i)^2$

---

## 9. Decision Boundary by Network Depth

### 🎯 Picture this

Cutting shapes out of paper:
- **0 hidden layers:** you may only make **one straight cut** → a line.
- **1 hidden layer:** several straight cuts that join into one **convex blob** (no dents).
- **2 hidden layers:** combine blobs freely → **any shape**, even separate islands.

### 🔗 Why this matters

| Hidden layers | Can represent | Boundary shape |
|---|---|---|
| **0** | Linear classifier | A single straight **hyperplane** |
| **1** | Convex region | Boundary of an **open/closed convex region** |
| **2** | Combinations | **Arbitrary** shapes (even disjoint) |

```
  0 hidden:        1 hidden:          2 hidden:
   ╲  (one line)   ╱──╲ (convex blob)  ╭──╮ ╭──╮ (any/disjoint shape)
    ╲              ╲──╱                ╰──╯ ╰──╯
```

> 🔑 **Takeaway:** depth = expressive power. 0 = only lines, 1 = convex blobs, 2 = essentially anything. This is *why* deep networks are powerful.

---

## 10. Activation Functions

### 🎯 Picture this

The activation is the "finishing process" at each assembly-line station — it decides *how* the combined signal gets shaped before moving on. Different finishes have different properties (range, smoothness, whether they're trainable).

### 🔗 Why this matters

| Name | Equation | Derivative | Note |
|---|---|---|---|
| **Identity** | $f(x)=x$ | $f'=1$ | Linear (no squashing) |
| **Binary step** | 0 if x<0, 1 if x≥0 | 0 (undef at 0) | Not differentiable → can't train |
| **Logistic (sigmoid)** | $\dfrac{1}{1+e^{-x}}$ | $f'=f(x)(1-f(x))$ | Smooth, output (0,1) |
| **TanH** | $\dfrac{2}{1+e^{-2x}}-1$ | $f'=1-f(x)^2$ | Output (−1,1), zero-centred |
| **ReLU** | 0 if x<0, x if x≥0 | 0 if x<0, 1 if x≥0 | Most popular in deep nets |
| **Leaky/PReLU** | αx if x<0, x if x≥0 | α if x<0, 1 if x≥0 | Fixes ReLU "dying" |
| **ELU** | α(eˣ−1) if x<0, x if x≥0 | — | Smooth ReLU variant |
| **SoftPlus** | $\log_e(1+e^x)$ | $\dfrac{1}{1+e^{-x}}$ | Smooth ReLU; deriv = sigmoid |

> 💡 **Remember the sigmoid derivative** $f'(x) = f(x)(1-f(x))$ — it reuses the forward output, making backprop very clean.

#### 🧠 Worked example — sigmoid + its derivative

At $x = 0$: $f(0) = \frac{1}{1+1} = 0.5$. Derivative $f'(0) = 0.5(1-0.5) = \mathbf{0.25}$ (the sigmoid's steepest point — slope max at the centre).

#### ✏️ Practice example (made up — solved)

Compute **ReLU** and its derivative at three inputs:

| $x$ | ReLU $= \max(0,x)$ | Derivative ($0$ if $x<0$, $1$ if $x\geq0$) |
|---|---|---|
| $-3$ | $0$ | $0$ (flat — "dead" here) |
| $0$ | $0$ | $1$ (by the $x\geq0$ rule) |
| $5$ | $5$ | $1$ |

Also the **sigmoid** at $x = 2$: $f(2) = \frac{1}{1+e^{-2}} \approx 0.88$, so $f'(2) = 0.88(1-0.88) = 0.88 \times 0.12 \approx \mathbf{0.106}$.
→ Note the sigmoid's slope (0.106) is much smaller out at $x=2$ than at its centre ($0.25$ at $x=0$) — it flattens at the edges (this flattening is the "vanishing gradient" issue ReLU was designed to avoid).

---

## 11. Training a Neural Network

### 🎯 Picture this

Same study loop as before: make a prediction, see how wrong, figure out *which knobs to turn and by how much*, turn them a little, repeat. For a network the "figure out which knobs" step is **backpropagation**; the "turn them" step is **gradient descent**.

### 🔗 Why this matters

```
t = 0;  max_iterations = 1000
Initialize θ₀ = [W₁⁰,…,W_L⁰, b₁⁰,…,b_L⁰]
while t++ < max_iterations:
    forward_propagation(θₜ)   ← compute the prediction
    ∇θₜ = backward_propagation(…)   ← compute gradients (which knobs, how much)
    θ_{t+1} = θₜ − η ∇θₜ            ← update (turn the knobs a little)
```

- **Forward propagation:** for each layer, $a_k = b_k + W_k h_{k-1}$, then $h_k = g(a_k)$, until $\hat y = O(a_L)$.
- **Backward propagation:** compute the output gradient $\nabla_{a_L}\mathcal{L} = -(e(y) - f(x))$, then propagate gradients **backward** through each layer.
- **Update:** $\theta_{t+1} = \theta_t - \eta\nabla\theta_t$ (same gradient-descent rule as Lecture 4; $\eta$ = learning rate).

---

## 12. Backpropagation

### 🎯 Picture this

A mistake gets made at the end of a long **chain of people passing a message** (telephone game). To assign blame fairly, you walk **backward** down the chain: "how much did the last person's change affect the final error? × how much did the previous person affect the last person? × …" Multiply the blame link-by-link, all the way back. That chain-of-multiplications is exactly the **chain rule** from calculus, and that's all backpropagation is.

### 🔗 Why this matters

> **Backpropagation = repeated application of the CHAIN RULE**, used to compute how much each weight contributed to the error.

#### The chain rule

Given $y = g(u)$ and $u = h(x)$:
$$\frac{dy_i}{dx_k} = \sum_{j=1}^{J}\frac{dy_i}{du_j}\cdot\frac{du_j}{dx_k}$$

> 💡 **Plain English:** to find how the final loss changes w.r.t. an early weight, multiply derivatives **link by link backward**: "how does loss change with $y$ × how does $y$ change with $a$ × how does $a$ change with the weight."

```
   FORWARD:   x ──► a ──► y ──► J (loss)     [predict & measure error]
   BACKWARD:  x ◄── a ◄── y ◄── J            [push blame back]
```

#### Case 1 — Logistic Regression (single neuron)

**Forward:** $a = \sum_{j=0}^{D}\theta_j x_j$, $y = \frac{1}{1+e^{-a}}$, $J = y^*\log y + (1-y^*)\log(1-y)$.

**Backward (chain rule):**
$$\frac{dJ}{dy} = \frac{y^*}{y} + \frac{(1-y^*)}{y-1}, \qquad \frac{dy}{da} = y(1-y), \qquad \frac{da}{d\theta_j} = x_j$$
$$\Rightarrow \frac{dJ}{d\theta_j} = \frac{dJ}{da}\cdot x_j$$

> 🔑 **Core backprop pattern:** gradient for weight $\theta_j$ = (error signal at $a$) × (the input $x_j$ that weight multiplied).

#### Case 2 — MLP (one hidden layer)

Graph: input → hidden linear $a_j=\sum\alpha_{ji}x_i$ → hidden sigmoid $z_j$ → output linear $b=\sum\beta_j z_j$ → output sigmoid $y$ → loss $J=\frac12(y-y^*)^2$.

Key links (each = "next gradient × local derivative"):
- $\dfrac{db}{d\beta_j} = z_j$ (output-weight gradient = hidden activation)
- $\dfrac{db}{dz_j} = \beta_j$ (push gradient back through output weights)
- $\dfrac{dz_j}{da_j} = z_j(1-z_j)$ (sigmoid derivative)
- $\dfrac{da_j}{d\alpha_{ji}} = x_i$ (hidden-weight gradient = input)

> 🔑 **All of backprop in one line:** *gradient w.r.t. a weight = (gradient flowing back to that neuron) × (the input that weight multiplied).* Everything else is chaining sigmoid-derivative ($z(1-z)$) and weight-passthrough ($\beta_j$) terms backward.

#### 🧠 Worked mini-example — chain rule numerically

Output $y = 0.8$, true $y^* = 1$, loss $J=\frac12(y-y^*)^2$.
- $\dfrac{dJ}{dy} = (y - y^*) = 0.8 - 1 = -0.2$
- sigmoid deriv $\dfrac{dy}{db} = y(1-y) = 0.8(0.2) = 0.16$
- $\dfrac{dJ}{db} = (-0.2)(0.16) = -0.032$
- hidden activation feeding it $z_j = 0.5$: $\dfrac{dJ}{d\beta_j} = (-0.032)(0.5) = \mathbf{-0.016}$

Then update $\beta_j^{new} = \beta_j - \eta(-0.016)$.

#### ✏️ Practice example (made up — solved)

Output $y = 0.6$, true $y^* = 0$, loss $J=\frac12(y-y^*)^2$. A hidden activation feeding the output is $z_j = 0.9$, learning rate $\eta = 0.1$, current $\beta_j = 0.4$.

**Walk the chain backward:**
1. $\dfrac{dJ}{dy} = (y - y^*) = 0.6 - 0 = 0.6$
2. sigmoid derivative $\dfrac{dy}{db} = y(1-y) = 0.6(1-0.6) = 0.6 \times 0.4 = 0.24$
3. $\dfrac{dJ}{db} = \dfrac{dJ}{dy}\cdot\dfrac{dy}{db} = (0.6)(0.24) = 0.144$
4. $\dfrac{dJ}{d\beta_j} = \dfrac{dJ}{db}\cdot z_j = (0.144)(0.9) = \mathbf{0.1296}$

**Update the weight:** $\beta_j^{new} = \beta_j - \eta\dfrac{dJ}{d\beta_j} = 0.4 - (0.1)(0.1296) = 0.4 - 0.01296 \approx \mathbf{0.387}$.

→ The weight shrank slightly because it contributed to an over-prediction (said 0.6 when truth was 0). Backprop does this for **every weight** automatically, link by link.

---

## 13. The Night-Before Recap

**Biological → ANN** (the decision desk): dendrite=input, synapse=weight, soma=process, axon=output.

**MP neuron (1943)** = hand-counting committee: $y=1$ if $\sum x_i \geq \theta$ else 0. No weights, Boolean. $y=0$ if any input inhibitory (veto). AND gate: θ=n. OR gate: θ=1. **Limitation:** single MP → only **linearly separable** functions (XOR fails).

**Perceptron** = committee where votes are weighted *and* learnable. Worked: $w=(2,-4,1)$, (1,0,0)→v=2→y=1. Practice: $w=(-1,3,2)$, (2,1,0)→v=1→y=1.

**MLP** = a layered team. **Rule:** any Boolean fn of $n$ inputs → 1 hidden layer with $2^n$ perceptrons + 1 output. ($n=3\to8$, $n=4\to16$.)

**Perceptron vs Sigmoid** = switch vs dimmer. Step = not differentiable (no slope to train on). Sigmoid = smooth, **differentiable** → enables backprop.

**FFNN equations** (the assembly line):
- Pre-activation: $a_i = b_i + W_i h_{i-1}$ (combine parts)
- Activation: $h_i = g(a_i)$ (finishing process)
- Output: $f(x)=h_L=O(a_L)$
(Worked: $h_0=(2,3)$ → $h_1=(0.95,0.95)$. Practice: $h_0=(1,2)$ → $h_1=(0.95,0.73)$.)

**Decision boundary by depth** (paper cuts): 0 hidden = line; 1 hidden = convex blob; 2 hidden = any/disjoint shape.

**Activations:** sigmoid $\frac{1}{1+e^{-x}}$, deriv $f(1-f)$ (0.25 at centre); tanh deriv $1-f^2$; ReLU $=\max(0,x)$; step not differentiable.

**Training loop:** forward → backward → $\theta_{t+1}=\theta_t-\eta\nabla\theta_t$.

**Backpropagation = repeated chain rule** (assigning blame backward down a chain). Pattern: *gradient w.r.t. a weight = (gradient flowing back to neuron) × (input that weight multiplied)*. Sigmoid deriv $=y(1-y)$ reused. Worked: y=0.8, y*=1 → dJ/dβ=−0.016. Practice: y=0.6, y*=0 → dJ/dβ≈0.13, β: 0.4→0.387.

---

*✅ Study guide for Lecture 5 (Artificial Neural Networks) complete.*
