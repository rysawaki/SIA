# Experiment Report 04: Temporal Evolution of Interpretation via Self-Formation (Recursive Reinforcement)

**Script:** `experiments/04_exp_attention_evolution.py` (Provisional)  
**Date:** 2025-11-29  
**Status:** ✅ Recursive Self-Reinforcement Observed

## 1. Objective
This experiment verified **how the AI's "Interpretation (Query)" and "Attention" change over time** as the Self accumulates experience, even when facing identical inputs.
Specifically, we executed the following recursive loop for 5 steps:

1.  **Perception:** Process input and generate a Query.
2.  **Experience:** Feed that Query (interpretation) back into the Self as a Trace.
3.  **Transformation:** The Self is reinforced and biases the Query generation in the next step.

This is a simulation of **Confirmation Bias** or **Fixation of Belief**, where "believing in one's own interpretation reinforces that perspective."

## 2. Analysis of Results

### ① Visualization of Query Trajectory

![Query Trajectory](image_828014.png)

* **How to read the plot:**
    * **Color:** Corresponds to each token (Token 0–5).
    * **● (Circle):** Start point (Step 0). No Self influence.
    * **× (Cross):** End point (Step 4). After repeated Self interventions.
    * **Line:** The trajectory of interpretation shift over time.
* **Analysis:**
    * It is evident that the Queries for all tokens are being **pulled toward a specific direction (the center of gravity formed by the Self)**.
    * In particular, **Token 0 (Dark Blue)** and **Token 2 (Cyan)** show long travel distances, deviating significantly from their initial interpretations. This implies that the Self has strongly "dyed" the meaning of these tokens with its own color.
    * **Conclusion:** The Self is not a static filter but functions as a **"Dynamic System"** that causes inputs to converge into a specific attractor (basin of attraction) over time.

### ② Evolution of Attention Map

![Attention Map Evolution](image_828034.png)

* **Analysis:**
    * **Step 0 (Far Left):** Initial Attention distribution.
    * **Step 4 (Far Right):** Distribution after Self intervention.
    * As we proceed from left to right, the heatmap pattern shifts (drifts) subtly but surely.
    * This indicates that although the input data is exactly the same, **the criteria for determining "what is important" have changed.**
    * A noticeable pattern change is observed around Step 2, suggesting that the focus of attention shifted once the accumulation of Self exceeded a certain threshold.

### ③ Evolution of Self Metrics (Console Output)

```text
[Step 1] Self Metrics: {'num_axes': 1, 'strength_sum': 0.50}
...
[Step 5] Self Metrics: {'num_axes': 1, 'strength_sum': 2.07}
```

* **Analysis of Self Metrics:**

    * **`num_axes`:** **Constant at 1.**
    *  Since the input remained identical throughout the steps, the Self did not perceive a need to create new axes (new values). Instead, it **continuously reinforced the single existing axis (the initial impression).**
    * **Increase in `strength_sum`:**
    * It surged from **0.50 to 2.07**. This confirms that the **influence (gravitational pull) of the Self became stronger with each step.**

## 3. Theoretical Discussion: Why did it converge?

In this experiment, we performed a loop where the model's own output (Query) is fed back into the Self as a Trace (similar to self-supervised learning). This created a feedback loop:

1.  **Initial State:** Initially, the model vaguely thinks "It looks like A" (Step 0).
2.  **Imprinting:** That "Interpretation of A" is stored in the Self.
3.  **Biasing:** The Self applies a bias that "It IS A."
4.  **Reinforcement:** Looking at it next time, it appears "Definitely A" (Step 1 and beyond).
5.  **Convergence:** This repeats, and the interpretation converges to a single point.

This phenomenon perfectly reproduces the **"Stabilization of Identity"** process in SIA theory, while also demonstrating the potential for **"Loss of Flexibility (Fixation)"** if carried too far.

## 4. Next Steps

In the current experiment, the axis converged to one direction because the model only looked at "itself" (recursive input). The following steps are recommended:

* **Control Experiment:** Introduce a **"totally different input (Surprise)"** midway through the process.
    * **Goal:** To verify whether this converged/fixated Self **rejects** the new input (cognitive dissonance) or **creates a new axis** to adapt (accommodation).