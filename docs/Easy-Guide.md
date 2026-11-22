# CardioAgent-Ops — The Easy Guide

Read this first. Then the big PDF (`CardioAgent-Ops-Defense-Guide.pdf`) will make sense.

---

## 1. What does the project do? (say this in 20 seconds)

> "My project listens to a heart-sound recording and says if it sounds **normal** or **abnormal**.
> Then it searches official heart guidelines (ESC) and an AI writes a short, careful explanation
> with real quotes. It is a school research project, **not a doctor**."

---

## 2. The 3 helpers (the "agents")

Think of 3 people working in a line:

| Helper | Real name | What it does |
|---|---|---|
| 🎧 The **Listener** | AudioAgent | Listens to the heart sound → "normal" or "abnormal" + a score |
| 📚 The **Librarian** | KnowledgeRAG-Agent | Finds useful paragraphs in the ESC guideline PDFs |
| ✍️ The **Writer** | ReasoningLLM-Agent | The AI (Mistral, on my own PC) writes a short report using the quotes |

Inside the Writer there is a **checker**: if the AI invents a quote or says something dangerous,
the report is **rejected**. Better no report than a wrong one.

---

## 3. The data (the heart sounds)

- Two public collections of heart sounds: **PhysioNet** and **PASCAL**.
- After cleaning (remove copies, bad quality, files without a label): **2,960 recordings**.
- Label: **0 = normal**, **1 = abnormal**.

We cut them like school:

| Part | % | Like… | Used for |
|---|---|---|---|
| Training | 60% (1,776) | the lessons | the model learns |
| Validation | 20% (592) | the practice exam | choose the best settings |
| Test | 20% (592) | the final exam | measured **once**, at the end |

Rule: **never study with the final exam**. We never used the test to choose anything.

---

## 4. How a sound becomes numbers

A computer cannot "hear". So:

1. Keep only the **first 5 seconds**.
2. Make it simple: 1 channel, 2,000 points per second.
3. Turn it into numbers:
   - **84 numbers** (a "fingerprint" of the sound) → for the simple models.
   - **A picture of the sound** (96×96 image, called a Mel spectrogram) → for the neural networks.

---

## 5. The 5 models (5 "students" taking the same exam)

| Model | In simple words |
|---|---|
| **Decision Tree** | A list of yes/no questions, like a flowchart |
| **SVM** | Draws the best line separating normal from abnormal |
| **CNN** | A neural network that looks at the sound picture |
| **ResNet50** | A big network trained on photos, re-used for sound pictures |
| **Transformer–CNN** | A CNN + "attention" to follow how the sound changes in time |

---

## 6. The 3 words you MUST understand

Imagine 125 sick hearts and 467 healthy hearts in the final exam.

- **Recall** = of the **sick** hearts, how many did we catch?
  *Missing a sick heart is the worst mistake.*
- **Specificity** = of the **healthy** hearts, how many did we correctly call healthy?
  *Avoids false alarms.*
- **Precision** = when we say "abnormal", how often are we right?

---

## 7. The results (final exam)

| Model | Catches sick hearts (Recall) | Correct on healthy (Specificity) |
|---|---|---|
| ⭐ **Decision Tree (chosen)** | **98.4%** | 71% |
| SVM | 76.8% | **95%** ← best at avoiding false alarms |
| CNN | 83.2% | 83% |
| ResNet50 | 76% | 94% |
| Transformer–CNN | 90.4% | 85% |

### Why the Decision Tree won
Our rule (fixed **before** the final exam):
> "Catch the most sick hearts, **but** you must be right on at least half of the healthy ones."

The Tree caught **123 of 125** sick hearts → only **2 missed**.

### Its weakness (say it honestly!)
It gave **135 false alarms** (healthy hearts called abnormal).
→ It is a **choice**: we prefer false alarms over missed sick hearts.

### Why the rule "at least half of healthy ones"?
A silly model that says "abnormal" to **everyone** would catch 100% of sick hearts…
but would be wrong on every healthy one. The rule blocks that cheat.

### ✅ Proof it is real
We re-trained the **Decision Tree and SVM** on my computer (same 2,960 recordings, same split)
and got **exactly the same numbers** (123 caught, 2 missed, 135 false alarms).
→ The work is **reproducible**. (The 3 neural networks were not retrained here: their saved
results come from the original 33-minute run.)

---

## 8. The "MLOps" tools (how it is organized like a real product)

| Tool | Simple meaning |
|---|---|
| **MLflow** | A lab notebook: saves every training, its settings and its scores |
| **ONNX** | Saves the trained model in a light, universal file to run anywhere |
| **Docker** | 6 small boxes, each with one job (website, agents, audio, AI, database, MLflow) |
| **Qdrant** | The database where guideline paragraphs are stored to be searched |
| **Ollama / Mistral** | The AI writer, running locally on my PC (no data sent outside) |

---

## 9. Honest limits (teachers LOVE hearing these)

1. It is **not a medical diagnosis**.
2. No patient IDs → I cannot prove training and test patients are different people.
3. The AI (Mistral) was **not fine-tuned** (the assignment asked for QLoRA — I say it openly).
4. The score is **not** a real "chance of disease".
5. Only the first 5 seconds of each sound are used.

---

## 10. Easy answers to likely questions

**"What did you build?"**
A system that classifies heart sounds, finds guideline evidence, and writes a checked report.

**"Why not let the AI do everything?"**
The audio model is a specialist for sound. The AI only writes text — it never hears the audio.

**"Why not just use accuracy?"**
A model saying "normal" to everyone looks accurate but misses every sick heart.
So we use recall and specificity.

**"Why the Tree and not the SVM?"**
Our priority is not missing sick hearts. The SVM is better at avoiding false alarms.
It is a choice of priorities, decided before the final exam.

**"Did you use the test to choose?"**
No. Choices were made on validation. Test was used once, at the end.

**"Is it ready for hospitals?"**
No. It needs real patient validation and doctors reviewing the reports first.

---

## If you remember only 3 things

1. **3 helpers**: Listener → Librarian → Writer (+ checker).
2. **Tree: 98.4% sick hearts caught, but 135 false alarms** — a deliberate trade-off.
3. **Research only, not a doctor** — and the results were reproduced on my PC.
