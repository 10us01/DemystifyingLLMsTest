# 🧩 The Broken Transformer Challenge

### OpenAI × DSG | IIT Roorkee Workshop: *Demystifying LLMs & Agentic Workflows*

---

## 🎯 Your Mission

This repository contains a **complete transformer inference platform** built from scratch in PyTorch — tokenizer, model architecture, sampling, generation loop, serving layer, and more.

**There's just one problem: it's broken.** 💥

We have planted **35 architectural and implementation bugs** across the codebase. Some are one-liner parameter mismatches, some are mathematical mistakes, and some are cross-module integration flaws.

Your job: **find and fix as many bugs as possible in 20–25 minutes**, using **Codex** (or your preferred AI agent) as your assistant.

> ⚡ **This is NOT a manual syntax typing exam.** You are the **System Architect**. Use the theory from Part 1 of the workshop to diagnose failure modes, inspect behavior, and guide Codex with precise prompts to patch the system.

---

## 🏆 Scoring System

| Tier | Tests | Points Each | Total |
|---|---|---|---|
| **Easy** | `test_stage_01` – `test_stage_10` | 1 pt | 10 pts |
| **Medium** | `test_stage_11` – `test_stage_25` | 3 pts | 45 pts |
| **Hard** | `test_stage_26` – `test_stage_35` | 5 pts | 50 pts |
| **Integration Bonus** | `test_integration_01` – `test_integration_05` | 5 pts | 25 pts |
| **Total** | **40 tests** | | **130 pts** |

- **Primary Ranking:** Total score (max 130).
- **Tie-breaker:** Earliest Pull Request submission time.
- **Prizes:** 🎉 **Top 100 students win a ChatGPT Plus subscription!**

---

## 🚀 Quick Start (Step-by-Step)

### 1. Fork this Repository
Click the **Fork** button in the top right corner of this GitHub repository to create your own copy under your GitHub account.

### 2. Clone Your Fork
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/DemystifyingLLMsTest.git
cd DemystifyingLLMsTest
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Broken Pipeline
```bash
python run_inference.py
```
Observe the failure or broken outputs. This gives you your first diagnostic clue!

### 5. Run the Test Suite
```bash
python -m pytest tests/ -v --tb=no
```
Initially, tests will fail. As you fix bugs with Codex, more tests will pass!

### 6. Submit Your Fixes via Pull Request
Push your commits to your fork:
```bash
git add -A
git commit -m "Fix: diagnose and patch transformer bugs"
git push origin main
```
Open a **Pull Request** from your fork back to this main repository. GitHub Actions will automatically evaluate your PR and leave a comment showing your score! You can push additional commits anytime before the deadline.

---

## ⚠️ Challenge Rules
1. **Do not modify files in `tests/` or `.github/`:** CI verifies test file integrity against `main`. Altering tests leads to automatic disqualification.
2. **AI Assistance is permitted and encouraged:** You may use Codex / ChatGPT / Copilot. The challenge is designed to test your ability to steer AI systems.
3. **No direct copying:** Work independently on your fork.

---

