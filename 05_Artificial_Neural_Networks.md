# Lecture 5: Artificial Neural Networks (ANN)

> **Course:** BITS F464 — Machine Learning
> **Covers:** Biological inspiration, McCulloch-Pitts neuron (+ Boolean gates, geometric interpretation), Perceptron & its learning algorithm, Sigmoid neuron, Multilayer Perceptron (MLP), Feed-Forward Neural Network (pre-activation/activation equations), decision boundaries by depth, activation functions, network training, and **Backpropagation** (the chain rule).
>
> **How to read this:** Every formula has a worked numerical example. The single-neuron calculation, AND/OR gate thresholds, the FFNN pre-activation, and backprop chain-rule are the exam-likely problems — all worked end-to-end.

> 📌 *Note:* Some of this lecture (Perceptron, sigmoid, gradient descent) repeats Lecture 4. Those parts are summarized briefly here with cross-references; **see [04_LogisticReg_DecisionTree_Metrics.md](04_LogisticReg_DecisionTree_Metrics.md) Part A** for the full treatment. The **new** material is MP neuron Boolean gates, MLP, FFNN equations, and backpropagation.

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
13. [Quick Revision Sheet](#13-quick-revision-sheet)

---

## 1. Biological Inspiration

The fundamental unit of an ANN is an **artificial neuron**, inspired by the **biological neuron** in the brain.

| Biological part | Job | ANN equivalent |
|---|---|---|
| **Dendrite** | Receives inputs from other neurons | Inputs $x_1, x_2, \dots$ |
| **Synapse** | Connection point to other neurons | Weights $w_1, w_2, \dots$ |
| **Soma** | Processes the information | The aggregation + activation |
| **Axon** | Transmits this neuron's output | Output $y$ |

```
   Biological:  dendrites → soma → axon → (synapse) → next neuron
   Artificial:  inputs xᵢ → Σ(wᵢxᵢ) → activation σ → output y
```

> 💡 **Memory hook:** **D**endrite = **D**ata in. **A**xon = **A**nswer out. Synapse = the *strength* of a connection = the weight.

---

## 2. McCulloch-Pitts (MP) Neuron

The **first computational model of a neuron (1943)** — very simplified.

- **$g$** = aggregates the inputs (just sums them).
- **$f$** = takes a decision based on that aggregation (threshold).
- Inputs can be **excitatory** (encourage firing) or **inhibitory** (block firing).
- **Rule: $y = 0$ if ANY input is inhibitory.** Otherwise:

$$g(x_1, \dots, x_n) = \sum_{i=1}^{n} x_i$$
$$y = f(g(\mathbf{x})) = \begin{cases} 1 & \text{if } g(\mathbf{x}) \geq \theta \\ 0 & \text{if } g(\mathbf{x}) < \theta \end{cases}$$

- **$\theta$ (theta) = the thresholding parameter** (also called **thresholding logic**).
- Inputs and output are **Boolean** ($\in \{0, 1\}$). **No weights** (that's the perceptron's later upgrade).

```
   x₁ ─┐
   x₂ ─┼──►( g = Σxᵢ )──►[ ≥ θ ? ]──► y ∈ {0,1}
   xₙ ─┘
```

#### 🧠 Worked example

3 inputs $(x_1,x_2,x_3) = (1,1,0)$, threshold $\theta = 2$, none inhibitory:
$$g = 1+1+0 = 2, \quad 2 \geq 2 \;\Rightarrow\; y = 1.$$
If $\theta = 3$: $g = 2 < 3 \Rightarrow y = 0$.

---

## 3. Boolean Functions using MP

By choosing the right threshold $\theta$, a single MP neuron implements logic gates. (Assume 3 inputs each 0/1.)

| Gate | Threshold $\theta$ | Fires (y=1) when |
|---|---|---|
| **AND** | **3** | ALL three inputs are 1 (sum = 3 ≥ 3) |
| **OR** | **1** | AT LEAST one input is 1 (sum ≥ 1) |

#### 🧠 Worked example — verify AND gate ($\theta = 3$)

| $x_1$ | $x_2$ | $x_3$ | sum | sum ≥ 3? | $y$ |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 3 | yes | **1** ✓ |
| 1 | 1 | 0 | 2 | no | **0** ✓ |
| 0 | 0 | 0 | 0 | no | **0** ✓ |

→ Only fires when *all* are on = exactly AND behaviour. ✓

#### 🧠 Worked example — verify OR gate ($\theta = 1$)

| $x_1$ | $x_2$ | $x_3$ | sum | sum ≥ 1? | $y$ |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | no | **0** ✓ |
| 1 | 0 | 0 | 1 | yes | **1** ✓ |
| 0 | 1 | 1 | 2 | yes | **1** ✓ |

→ Fires when *any* is on = OR behaviour. ✓

> 🔑 **Exam tip:** For an $n$-input AND, $\theta = n$. For OR, $\theta = 1$. (You may be asked to design a gate by choosing θ.)

---

## 4. Geometric Interpretation & Limitation of MP

Think of inputs as **points in space**. The condition $\sum x_i = \theta$ is a **line** (in 2D) or **plane** (in 3D) that splits the space.

```
   OR (θ=1):  x₁+x₂ ≥ 1            AND (θ=2):  x₁+x₂ ≥ 2
   x₂                              x₂
   1 ●────────● (1,1)→1            1 ○        ● (1,1)→1
     │╲                              │      ╲
     │ ╲  line x₁+x₂=1               │       ╲ line x₁+x₂=2
   0 ○──╲─────● (1,0)→1            0 ○────────○
     (0,0)→0    x₁                  (0,0)→0  (1,0)→0  x₁
   ● = output 1,  ○ = output 0
```

> 🔑 **LIMITATION (very common exam question):** A **single MP neuron can only represent Boolean functions that are LINEARLY SEPARABLE.**

**Linearly separable** = there exists a single straight line (or plane/hyperplane) such that all points giving output **1 lie on one side** and all points giving output **0 lie on the other side**.

> ⚠️ AND, OR are linearly separable (one line works). **XOR is NOT** — no single line can separate it. This is the famous limitation that motivated **multilayer** networks (§6).

---

## 5. Perceptron (recap + worked example)

> 📌 Full treatment in [Lecture 4 Part A](04_LogisticReg_DecisionTree_Metrics.md). Brief recap here:

The **Perceptron** (Rosenblatt 1958; refined Minsky & Papert 1969) upgrades the MP neuron by **adding learnable weights** $w_i$. Inputs **no longer Boolean**.

$$y = 1 \text{ if } \sum_{i=0}^{n} w_i x_i \geq 0, \quad \text{else } 0 \qquad (x_0=1,\; w_0=-\theta)$$

**Perceptron learning:** misclassified positive → $\mathbf{w}=\mathbf{w}+\mathbf{x}$; misclassified negative → $\mathbf{w}=\mathbf{w}-\mathbf{x}$. Converges only if **linearly separable**.

#### 🧠 Worked example — single artificial neuron (slide 10, exam-likely)

A neuron with 3 inputs, weights $w_1=2,\; w_2=-4,\; w_3=1$, activation
$$\varphi(v) = \begin{cases} 1 & \text{if } v \geq 0 \\ 0 & \text{otherwise} \end{cases}, \qquad v = w_1x_1 + w_2x_2 + w_3x_3$$

**Pattern P1:** input $(x_1,x_2,x_3) = (1, 0, 0)$
$$v = 2(1) - 4(0) + 1(0) = 2 \quad (2 > 0) \;\Rightarrow\; y = \varphi(2) = \mathbf{1}$$

**Pattern P2:** input $(x_1,x_2,x_3) = (0, 1, 1)$
$$v = 2(0) - 4(1) + 1(1) = -3 \quad (-3 < 0) \;\Rightarrow\; y = \varphi(-3) = \mathbf{0}$$

> ✅ **Exam recipe:** ① compute $v = \sum w_i x_i$ → ② apply activation: $v \geq 0$ → 1, else 0. This *exact* problem style appears often.

---

## 6. Multilayer Perceptron (MLP)

> An **MLP** stacks neurons into **layers**, so it can solve problems a single neuron can't (like XOR / non-linearly-separable data).

**Example 3-layer network (slide 11):**

```
        y          ← Output layer: 1 neuron
       ╱│╲╲
   w₁ w₂ w₃ w₄     ← Layer-2 weights
   h₁ h₂ h₃ h₄     ← Hidden layer: 4 perceptrons (outputs h₁..h₄)
    ╲╲╳╳╱╱
   x₁    x₂         ← Input layer: 2 inputs
   (Layer-1 weights: red edge w=−1, blue edge w=+1; bias=−2)
```

| Layer | Contents |
|---|---|
| **Input layer** | $x_1, x_2$ (the raw features) |
| **Hidden layer** | 4 perceptrons, producing $h_1, h_2, h_3, h_4$ |
| **Output layer** | 1 neuron → final $y$ |

> 🔑 **The "simple rule" (exam fact):** *Any Boolean function of $n$ inputs can be represented by a network with **1 hidden layer containing $2^n$ perceptrons** and **1 output perceptron**.*

#### 🧠 Worked example — applying the rule

For $n = 3$ inputs: hidden layer needs $2^n = 2^3 = \mathbf{8}$ perceptrons + 1 output perceptron (slide 12's diagram shows exactly 8 hidden units, bias = −3).

> 💡 **Why $2^n$?** A Boolean function of $n$ inputs has $2^n$ possible input combinations. In the worst case you need one hidden neuron to "detect" each specific combination, then OR them at the output.

---

## 7. Perceptron vs Sigmoid Neuron

The perceptron's **step function** has a problem: it's a hard jump.

| | **Perceptron (step)** | **Sigmoid (logistic) neuron** |
|---|---|---|
| Output rule | $y=1$ if $\sum w_ix_i \geq 0$ else 0 | $y = \dfrac{1}{1+e^{-\sum w_ix_i}}$ |
| Shape | Hard step (0 → jumps to 1) | Smooth S-curve |
| Properties | **Not smooth, not continuous, NOT differentiable** | **Smooth, continuous, DIFFERENTIABLE** |

```
   PERCEPTRON (step)            SIGMOID
 y │     ┌──────             y │       ╭─────
 1 │     │ hard jump         1 │    ╭──╯ smooth
 0 │─────┘                   0 │──╮─╯
   └─────┼──►                  └────┼──►
```

> 🔑 **Why differentiability matters (critical exam point):** training uses **gradient descent**, which needs **derivatives**. The step function isn't differentiable → can't compute gradients → **can't train deep networks**. The sigmoid IS differentiable → enables **backpropagation**. *This single fact is why neural networks use sigmoid/smooth activations, not step functions.*

---

## 8. Feed-Forward Neural Network (FFNN)

A **special case of MLP** where data flows strictly **forward** (input → hidden → output, no loops).

### 8.1 The setup & notation

| Symbol | Meaning |
|---|---|
| Input layer | N-dimensional vector; called the **0th layer** |
| Hidden layers | $L-1$ hidden layers, each with $n$ neurons |
| Output layer | $k$ neurons (one per class); called the **Lth layer** |
| $a_i$ | **pre-activation** at layer $i$ (the linear part, before squashing) |
| $h_i$ | **activation** at layer $i$ (after applying activation function) |
| $W_i, b_i$ | weight & bias between layers $i{-}1$ and $i$ |

### 8.2 The two key equations (MUST memorize)

**Pre-activation** (linear combination — like the perceptron's $v$):
$$a_i(x) = b_i + W_i\, h_{i-1}(x)$$

**Activation** (apply non-linear function $g$):
$$h_i(x) = g\big(a_i(x)\big)$$

where $g$ = activation function (logistic/tanh/linear/etc.). The final output uses an output activation $O$ (softmax/linear):
$$f(x) = h_L(x) = O\big(a_L(x)\big)$$

```
  x ─► [a₁ = b₁ + W₁x] ─► [h₁ = g(a₁)] ─► [a₂ = b₂ + W₂h₁] ─► [h₂ = g(a₂)] ─► … ─► ŷ
       └─ pre-activation ┘ └─ activation ┘   (repeat per layer)
```

> 💡 **Two-step rhythm per layer:** ① **linear**: weighted sum + bias → $a$. ② **non-linear**: squash through $g$ → $h$. That $h$ becomes the *input* to the next layer. Repeat.

#### 🧠 Worked example — one layer's pre-activation & activation

Inputs $h_0 = (x_1, x_2) = (2, 3)$. Weights $W_1 = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix}$, bias $b_1 = (1, -1)$. Activation = sigmoid.

**Pre-activation $a_1 = b_1 + W_1 h_0$:**
- Neuron 1: $a_{11} = 1 + (1\cdot 2 + 0\cdot 3) = 1 + 2 = 3$
- Neuron 2: $a_{12} = -1 + (-1\cdot 2 + 2\cdot 3) = -1 + 4 = 3$

**Activation $h_1 = \sigma(a_1)$:**
$$h_{11} = \sigma(3) = \frac{1}{1+e^{-3}} = \frac{1}{1.0498} \approx 0.95$$
$$h_{12} = \sigma(3) \approx 0.95$$

→ This $h_1 = (0.95, 0.95)$ now feeds the next layer. Repeat the same two steps until the output.

### 8.3 Supervised learning in FFNN (slide 18)

- **Data:** $\{x_i, y_i\}_{i=1}^{N}$
- **Model:** $\hat{y}_i = O(W^3 g(W^2 g(W^1 x + b_1) + b_2) + b_3)$ (nested layers)
- **Parameters:** $\theta = W_1,\dots,W_L,\; b_1,\dots,b_L$
- **Algorithm:** Gradient Descent **with Backpropagation**
- **Loss:** e.g. $\min \dfrac{1}{N}\sum (\hat y_i - y_i)^2$, generally $\min \mathcal{L}(\theta)$

---

## 9. Decision Boundary by Network Depth

**This is a guaranteed conceptual exam question.** More hidden layers → more complex shapes the network can carve out.

| Hidden layers | What it can represent | Boundary shape |
|---|---|---|
| **0 hidden** | Linear classifier | A single straight **hyperplane** (line) |
| **1 hidden** | Convex region | Boundary of an **open or closed convex region** |
| **2 hidden** | Combinations of convex regions | **Arbitrary shapes** (any combination, even disjoint regions) |

```
  0 hidden:        1 hidden:          2 hidden:
   ╲  (one          ╱──╲  (convex      ╭──╮  ╭──╮  (any
    ╲  line)        │    │  blob)      │  │  │  │   complex /
     ╲              ╲──╱               ╰──╯  ╰──╯   disjoint shape)
```

> 🔑 **Takeaway:** depth = expressive power. 0 layers = only lines. 1 layer = convex blobs. 2 layers = essentially **anything**. This is *why* deep networks are powerful.

---

## 10. Activation Functions

The activation $g$ is what gives the network its non-linearity. Key ones from the slide table:

| Name | Equation | Derivative | Note |
|---|---|---|---|
| **Identity** | $f(x)=x$ | $f'=1$ | Linear (no squashing) |
| **Binary step** | 0 if x<0, 1 if x≥0 | 0 (undefined at 0) | Not differentiable → can't train |
| **Logistic (sigmoid)** | $\dfrac{1}{1+e^{-x}}$ | $f'=f(x)(1-f(x))$ | Smooth, output (0,1) |
| **TanH** | $\dfrac{2}{1+e^{-2x}}-1$ | $f'=1-f(x)^2$ | Output (−1,1), zero-centred |
| **ReLU** | 0 if x<0, x if x≥0 | 0 if x<0, 1 if x≥0 | Most popular in deep nets |
| **Leaky/PReLU** | αx if x<0, x if x≥0 | α if x<0, 1 if x≥0 | Fixes ReLU "dying" |
| **ELU** | α(eˣ−1) if x<0, x if x≥0 | — | Smooth ReLU variant |
| **SoftPlus** | $\log_e(1+e^x)$ | $\dfrac{1}{1+e^{-x}}$ | Smooth ReLU; deriv = sigmoid |

> 💡 **Memorize the sigmoid derivative** $f'(x) = f(x)(1-f(x))$ — it makes backprop calculations very clean (you reuse the forward output).

#### 🧠 Worked example — sigmoid + its derivative

At $x = 0$: $f(0) = \frac{1}{1+1} = 0.5$. Derivative $f'(0) = 0.5(1-0.5) = \mathbf{0.25}$ (the sigmoid's steepest point — its slope is maximum at the centre).

---

## 11. Training a Neural Network

The training loop (slide 23) is **gradient descent + backpropagation**:

```
t = 0;  max_iterations = 1000
Initialize θ₀ = [W₁⁰,…,W_L⁰, b₁⁰,…,b_L⁰]
while t++ < max_iterations:
    h₁,…,h_{L-1}, a₁,…,a_L, ŷ  =  forward_propagation(θₜ)   ← compute prediction
    ∇θₜ  =  backward_propagation(…)                          ← compute gradients
    θ_{t+1}  =  θₜ − η ∇θₜ                                   ← update (gradient descent)
```

- **Forward propagation** (slide 24): for each layer, $a_k = b_k + W_k h_{k-1}$, then $h_k = g(a_k)$, until $\hat y = O(a_L)$.
- **Backward propagation** (slide 24): compute the **output gradient** $\nabla_{a_L}\mathcal{L} = -(e(y) - f(x))$, then propagate gradients **backward** through each layer to get $\nabla_{W_k}$, $\nabla_{b_k}$.
- **Update rule:** $\theta_{t+1} = \theta_t - \eta\nabla\theta_t$ (same gradient-descent rule as [Lecture 4](04_LogisticReg_DecisionTree_Metrics.md); $\eta$ = learning rate).

---

## 12. Backpropagation

> **Backpropagation = just repeated application of the CHAIN RULE from calculus**, used to compute how much each weight contributed to the error.

### 12.1 The chain rule

Given $y = g(u)$ and $u = h(x)$:

$$\frac{dy_i}{dx_k} = \sum_{j=1}^{J}\frac{dy_i}{du_j}\cdot\frac{du_j}{dx_k}$$

> 💡 **Plain English:** to find how the final loss changes w.r.t. an early weight, multiply the derivatives **link by link backward** through the network. "How does loss change with $y$? × how does $y$ change with $a$? × how does $a$ change with the weight?" — chained together.

### 12.2 Why "backward"?

```
   FORWARD:   x ──► a ──► y ──► J (loss)     [compute prediction & error]
   BACKWARD:  x ◄── a ◄── y ◄── J            [push error gradients back]
              dJ/dx ← dJ/da ← dJ/dy ← J
```

You compute the error at the output, then **propagate the gradient backward** layer by layer (output → hidden → input), updating each weight by how much it contributed to the error.

### 12.3 Case 1 — Logistic Regression (single neuron, slide 26)

**Forward:**
$$a = \sum_{j=0}^{D}\theta_j x_j, \qquad y = \frac{1}{1+e^{-a}}, \qquad J = y^*\log y + (1-y^*)\log(1-y)$$
($y^*$ = true label, $y$ = prediction.)

**Backward (chain rule, step by step):**
$$\frac{dJ}{dy} = \frac{y^*}{y} + \frac{(1-y^*)}{y-1}$$
$$\frac{dJ}{da} = \frac{dJ}{dy}\cdot\frac{dy}{da}, \qquad \frac{dy}{da} = \frac{e^{-a}}{(e^{-a}+1)^2} \;(= y(1-y))$$
$$\frac{dJ}{d\theta_j} = \frac{dJ}{da}\cdot\frac{da}{d\theta_j}, \qquad \frac{da}{d\theta_j} = x_j$$

> 🔑 So $\dfrac{dJ}{d\theta_j} = \dfrac{dJ}{da}\cdot x_j$ — the gradient for weight $\theta_j$ is "(error signal at $a$) × (the input $x_j$ that weight multiplied)". This is the **core backprop pattern**.

### 12.4 Case 2 — MLP (one hidden layer, slide 27–28)

The computation graph (slide 27): **(A)** input → **(B)** hidden linear $a_j=\sum\alpha_{ji}x_i$ → **(C)** hidden sigmoid $z_j=\frac{1}{1+e^{-a_j}}$ → **(D)** output linear $b=\sum\beta_j z_j$ → **(E)** output sigmoid $y=\frac{1}{1+e^{-b}}$ → **(F)** loss $J=\frac12(y-y^*)^2$.

Backprop applies the chain rule through **every** stage backward:
$$\frac{dJ}{dy}\to\frac{dJ}{db}\to\underbrace{\frac{dJ}{d\beta_j}}_{\text{output weights}}\to\frac{dJ}{dz_j}\to\frac{dJ}{da_j}\to\underbrace{\frac{dJ}{d\alpha_{ji}}}_{\text{hidden weights}}$$

with the key links (each is "next gradient × local derivative"):
- $\dfrac{db}{d\beta_j} = z_j$  (output-weight gradient = hidden activation)
- $\dfrac{db}{dz_j} = \beta_j$  (push gradient back through output weights)
- $\dfrac{dz_j}{da_j} = \dfrac{e^{-a_j}}{(e^{-a_j}+1)^2} = z_j(1-z_j)$  (sigmoid derivative)
- $\dfrac{da_j}{d\alpha_{ji}} = x_i$  (hidden-weight gradient = input)

> 🔑 **The repeating pattern (this is ALL of backprop):** *gradient w.r.t. a weight = (gradient flowing back to that neuron) × (the input that weight multiplied)*. Everything else is just chaining sigmoid-derivative ($z(1-z)$) and weight-passthrough ($\beta_j$) terms backward.

#### 🧠 Worked mini-example — chain rule numerically

Say at the output: $y = 0.8$, true $y^* = 1$, loss $J=\frac12(y-y^*)^2$.
- $\dfrac{dJ}{dy} = (y - y^*) = 0.8 - 1 = -0.2$
- sigmoid deriv $\dfrac{dy}{db} = y(1-y) = 0.8(0.2) = 0.16$
- $\dfrac{dJ}{db} = (-0.2)(0.16) = -0.032$
- if a hidden activation feeding it is $z_j = 0.5$: $\dfrac{dJ}{d\beta_j} = \dfrac{dJ}{db}\cdot z_j = (-0.032)(0.5) = \mathbf{-0.016}$

Then update: $\beta_j^{new} = \beta_j - \eta(-0.016)$. That's one weight, one step. Backprop does this for **every weight** automatically via the chain rule.

---

## 13. Quick Revision Sheet

**Biological → ANN:** dendrite=input, synapse=weight, soma=process, axon=output.

**MP neuron (1943):** $y=1$ if $\sum x_i \geq \theta$ else 0. No weights, Boolean. $y=0$ if any input inhibitory.
- AND gate: θ=n (here 3). OR gate: θ=1.
- **Limitation:** single MP neuron → only **linearly separable** functions (XOR fails).

**Perceptron:** adds learnable weights; non-Boolean inputs. Worked: $w=(2,-4,1)$, input (1,0,0) → v=2 → y=1; input (0,1,1) → v=−3 → y=0.

**MLP:** layers of neurons solves non-linear problems. **Rule:** any Boolean fn of $n$ inputs → 1 hidden layer with $2^n$ perceptrons + 1 output.

**Perceptron vs Sigmoid:** step = not differentiable (can't train). Sigmoid = smooth, **differentiable** → enables backprop. *This is why we use sigmoid.*

**FFNN equations (memorize):**
- Pre-activation: $a_i = b_i + W_i h_{i-1}$ (linear)
- Activation: $h_i = g(a_i)$ (non-linear)
- Output: $f(x)=h_L=O(a_L)$

**Decision boundary by depth:** 0 hidden = line/hyperplane; 1 hidden = convex region; 2 hidden = any/complex shape.

**Activations:** sigmoid $\frac{1}{1+e^{-x}}$, deriv $f(1-f)$; tanh deriv $1-f^2$; ReLU = max(0,x). Step not differentiable.

**Training:** loop = forward_propagation → backward_propagation → $\theta_{t+1}=\theta_t-\eta\nabla\theta_t$.

**Backpropagation = repeated chain rule.** Pattern: *gradient w.r.t. a weight = (gradient flowing back to neuron) × (input that weight multiplied)*. Sigmoid deriv $= y(1-y)$ reused. Worked: y=0.8, y*=1 → dJ/db=−0.032 → dJ/dβ=−0.016.

---

*✅ Study guide for Lecture 5 (Artificial Neural Networks) complete. One PPT left — send the last one whenever ready.*
