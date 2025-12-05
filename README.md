# **CUCB-OTA: AI-Driven Real-Time Customer–Agent Routing**

### *🏆 **Hackotsava 2025 – National Level Hackathon Winner***

### *By Team Fullstack Alchemists*

---

## 🚀 **Overview**

CUCB-OTA (Constrained Upper Confidence Bound – Optimal Task Assignment) is an **AI-powered customer–agent routing engine** designed for real-time call centers and support systems.

It combines:

✔ **Causal ML (X-Learner)** for CSAT uplift
✔ **Optimal Assignment Algorithms** (Hungarian / Greedy)
✔ **Lagrangian Dual Optimization** for SLA, AHT & fairness constraints
✔ **Continuous feedback learning** (online optimization)

This system ensures **customers get the best agent**, **agents get balanced workload**, and **business constraints are never violated**.

---

# 🏆 **🏆 Achievement**

### **Winner – Hackotsava 2025 (National Level Hackathon)**

Our team **Fullstack Alchemists** won at **Hackotsava 2025**, competing against **87+ teams** across the country.
This solution stood out for:

🔹 Novel causal uplift modeling
🔹 Real-time routing efficiency
🔹 Optimal assignment using mathematical optimization
🔹 Clean architecture & scalability
🔹 Accurate constraint balancing using Lagrangian dual approach

---

# 📌 **Features**

### **1. Causal Uplift Modeling**

Predicts *how much more satisfied* a customer becomes with each agent.

### **2. Optimal Routing Engine**

* Hungarian algorithm → **Guaranteed optimal matching**
* Greedy fallback → **Fast for large batches**

### **3. Constraint-Aware Optimization**

System respects:

* SLA limits
* AHT budget
* Workload fairness
* Business rules

### **4. Self-Learning Feedback Loop**

Assignments → Outcomes → Penalty Updates → Smarter decisions next batch.

### **5. Fully Modular Architecture**

All components separated:

* `uplift_model.py` – X-Learner implementation
* `assignment.py` – Hungarian & Greedy routing
* `scoring.py` – Routing scores + constraints
* `config.py` – Central config management

---

# 🧠 **Architecture Diagram**

```
┌──────────────────────────────────────────────────────────────┐
│                     CUCB-OTA Workflow                         │
├──────────────────────────────────────────────────────────────┤
│ Historical Data → X-Learner → Uplift Predictions             │
│         ↓                                ↓                   │
│ New Customer Batch → Score Matrix → Hungarian Assignment     │
│         ↓                                ↓                   │
│ Feedback (CSAT, AHT, SLA) → Dual Penalties → Next Batch     │
└──────────────────────────────────────────────────────────────┘
```

---

# 📘 **Why This Approach?**

### ✨ **Causal > Correlation**

Instead of predicting *CSAT*, we predict **uplift** → “Which agent will improve CSAT the most?”

### ✨ **Optimal > Heuristic**

Hungarian ensures **mathematically optimal routing**, not guesswork.

### ✨ **Soft Constraints > Hard Constraints**

Lagrangian penalties dynamically adjust based on:

* SLA violations
* AHT overshoots
* Fairness drift

---

# 🧮 **Mathematical Optimization**

We maximize **total CSAT uplift**:

```
maximize   Σ τ(c,a) × x(c,a)
subject to:
   Σ AHT(c,a) × x(c,a) ≤ AHT_budget
   Σ SLA(c,a) × x(c,a) ≤ SLA_budget
   Gini(workload) ≤ fairness_budget
   Σ_a x(c,a) = 1  (each customer assigned once)
   x(c,a) ∈ {0,1}
```

---

# 📊 **Complexity Analysis**

| Component    | Time Complexity | Space  |
| ------------ | --------------- | ------ |
| X-Learner    | O(N·d·logN)     | O(N·d) |
| Score Matrix | O(K·M)          | O(K·M) |
| Hungarian    | O(n³)           | O(n²)  |
| Greedy       | O(n log n)      | O(n)   |
| Dual Update  | O(1)            | O(1)   |

---

# ⚙️ **Installation**

```bash
git clone https://github.com/<your-repo>
cd CUCB-OTA
pip install -r requirements.txt
```

---

# ▶️ **Quick Start**

```bash
python main.py
```

Or validate dependencies in 5 seconds:

```bash
python -c "
from config import config
print('✓ Config loaded')
"
```

---

# 📈 **Benchmarking**

| Batch | Agents | Raw (s) | Optimized (s) | Speedup   |
| ----- | ------ | ------- | ------------- | --------- |
| 50    | 30     | ~15.0   | ~0.4          | **37.5x** |
| 100   | 50     | ~45.0   | ~1.2          | **37.5x** |

---

# 🧩 **Limitations & Future Work**

❌ Currently uses synthetic data
✔ Add real CC data integration

❌ Single objective
✔ Add multi-objective (CSAT + revenue + retention)

❌ Batch processing only
✔ Add realtime streaming mode

---

# 🤝 **Team – Fullstack Alchemists**

* Pavan Kumar
* Sathwik K Bhat
* Swanjith
* Deepak


🏆 **Hackotsava 2025 Winners — Tetherfi Problem Statement**

---

# 📬 **Support**

For queries or contributions:

📧 **[pavankumar797524@gmail.com](mailto:pavankumar797524@gmail.com)**
![WhatsApp Image 2025-11-05 at 11 59 25_8fd74fa5](https://github.com/user-attachments/assets/8f323954-d908-4225-bb6d-0aec7cb5de22)

