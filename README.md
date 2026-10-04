# WEAVE: Web Evidence-Assisted Video Examination

> **From Web Evidence to Compact Verifiers: Distilling Verdicts and Rationales for Short-Video Misinformation Detection**

Research code for **WEAVE (Web Evidence-Assisted Video Examination)**, a framework for evidence-grounded short-video misinformation detection that transfers the final **verdict–rationale generation** task to a compact language model.

WEAVE reconstructs claims from short-video content, organizes multimodal and Web-assisted information into four source-aware evidence fields, uses a large reasoning teacher to generate offline supervision, and adapts a compact language model to produce both the final misinformation verdict and an inspectable rationale.

---

## Paper

**Full paper:** [weave_bigdata26.pdf](./weave_bigdata26.pdf)

### Authors

| Author | Role | Affiliation |
|---|---|---|
| **Chun-Yi Shih*** | Student | National Cheng Kung University, Tainan, Taiwan |
| **Kai-Yun Hsiao*** | Student | National Cheng Kung University, Tainan, Taiwan |
| **Yi-Hsien Tsai*** | Student | National Yang Ming Chiao Tung University, Hsinchu, Taiwan |
| **Cheng-Te Li** | Advisor | National Cheng Kung University, Tainan, Taiwan |

\* Chun-Yi Shih, Kai-Yun Hsiao, and Yi-Hsien Tsai contributed equally to this work.

### Keywords

`short-video misinformation` · `Web evidence` · `multimodal verification` · `knowledge distillation` · `compact language models`

---

## Overview

A genuine short video can spread misinformation when its caption changes the event, place, time, or surrounding context. Fact-checking therefore requires more than recognizing objects or detecting visual manipulation: the verifier must recover the claim, identify relevant event evidence, and explain how the evidence supports or contradicts that claim.

**WEAVE** addresses this problem through a claim-centered evidence interface and compact verdict–rationale generation.

The framework separates:

- reconstructed textual claims;
- video accounts;
- model-generated background knowledge;
- Web-assisted external context.

A large reasoning teacher generates offline verdict–rationale targets. Teacher-correct responses are used to supervise a compact language model, which is adapted using QLoRA to generate both the final verdict and its rationale.

---

## Framework

WEAVE separates online evidence preparation from offline distillation.

```text
Short Video + Title / Description
                |
                v
       Multimodal Content Recovery
        /                    \
       /                      \
Whisper Large V3       Visual Processing
Speech Transcript      Representative Frames
                       + Qwen3-VL Description
       \                      /
        \                    /
                |
                v
     Web-Assisted Evidence Preparation
       Gemma 4 31B IT + Google Search
                |
                v
       Four-Field Evidence Interface
       +--------+--------+--------+
       |        |        |        |
      Rc       Rv      K_int    K_ext
       |        |        |        |
       +--------+--------+--------+
                |
        +-------+-------+
        |               |
        v               v
 Offline Distillation   Online Inference
        |               |
 Teacher Verdict +      Compact WEAVE
 Rationale              Verifier
        |               |
 Teacher-Correct        Verdict + Rationale
 Supervision
        |
        v
 QLoRA Adaptation
```

The central idea is that **evidence preparation and final judgment are different tasks**. Large models recover and organize relevant evidence, while the compact verifier performs the final evidence comparison.

---

## Evidence Interface

For each video, WEAVE constructs four text fields:

| Field | Meaning | Role |
|---|---|---|
| `Rc` | Reconstructed textual claim | Recovers assertions expressed through the title and speech |
| `Rv` | Video account | Preserves visible claims, actions, entities, locations, dates, and event observations |
| `K_int` | Internal background | Provides relevant model-generated background information |
| `K_ext` | Web-assisted context | Provides search-assisted external event information |

The evidence record is conceptually:

```text
x_i = [Rc_i, Rv_i, K_int,i, K_ext,i]
```

The compact verifier receives these fields together with the verification instruction and generates:

```json
{
  "pred_label": 1,
  "reason": "The stated claim conflicts with the event information in the supplied evidence because ..."
}
```

Label convention:

```text
0 = real
1 = fake / misleading
```

---

## Method

### 1. Multimodal Content Representation

WEAVE accepts a short video and its accompanying title or description.

#### Speech Transcription

Audio is transcribed using **Whisper Large V3**. If the video has no usable audio or transcription fails, the transcript is treated as an empty string and the other inputs continue through the pipeline.

#### Representative Frames

Up to **16 representative frames** are distributed over the video duration. This view provides the evidence-preparation model with direct access to visual details throughout the clip.

#### Chronological Visual Description

A second visual representation is produced with **Qwen3-VL-4B-Instruct**.

Paper configuration:

| Setting | Value |
|---|---|
| Visual model | Qwen3-VL-4B-Instruct |
| Sampling | 5 FPS |
| Maximum sampled frames | 120 |
| Image resolution | Dynamic |
| Quantization | 4-bit NF4 |
| Example hardware | RTX 5070 Laptop GPU, 8 GB |

The representative frames and chronological description are complementary: frames preserve local visual details, while the generated description summarizes event order.

---

### 2. Web-Assisted Evidence Preparation

The paper uses **Gemma 4 31B IT** through the Gemini API with **temperature 0** and **Google Search enabled**.

A single external-model preparation request constructs the four evidence fields.

#### `Rc`: Reconstructed Textual Claim

Constructed from:

- title or description;
- speech transcript.

Its purpose is to recover the explicit textual assertion rather than provide a generic summary.

#### `Rv`: Video Account

Constructed from:

- `Rc`;
- chronological visual description;
- ordered representative frames.

It preserves verification-relevant details such as visible claims, entities, actions, locations, dates, temporal cues, and event order.

#### `K_int`: Internal Background

Contains relevant background supplied from the model's parametric knowledge.

#### `K_ext`: Web-Assisted Context

Contains information obtained through search-assisted verification. Searches target the specific event using combinations of entities, actions, locations, dates, and event-specific terminology.

Search queries and source URLs are retained as provenance metadata outside the four-field student input.

---

### 3. Claim–Evidence Verification

The verifier compares the assertion carried by the video against the supplied evidence.

The comparison includes checks for:

- reused footage with changed context;
- reassigned locations or settings;
- incorrect entities;
- incorrect dates or precise details;
- attachment of an unrelated narrative;
- physical implausibility;
- contradictions between textual claims and visible events.

Important evidence rules include:

- use both internal and external background information;
- do not infer falsity only because search results are missing;
- do not invent absent dates or details and then use them as contradictions;
- explain the decisive agreement or discrepancy rather than simply restating the predicted label.

---

### 4. Teacher Response Generation

The large-model teacher receives the four-field evidence record and the finalized verification instruction.

For each example, it generates:

```text
(predicted label, rationale)
```

These responses are used as offline supervision for the compact verifier.

The student therefore learns from **teacher-generated structured responses**, rather than teacher logits.

---

### 5. DSPy / MIPROv2 Prompt Development

**DSPy** with **MIPROv2** is used during development of the verification instruction and demonstrations.

This stage occurs after the evidence fields have been prepared and is not executed by the compact student during inference.

---

### 6. Teacher-Correct Supervision Selection

Let `I` denote examples with aligned evidence records, ground-truth labels, and teacher responses.

The teacher-correct pool is:

```text
C = { i in I : teacher_prediction_i = ground_truth_i }
```

`C` is split with class stratification into:

- training set `D_tr`;
- validation set `D_val`;
- held-out teacher-correct set `E1`.

Teacher-error examples form:

```text
E2 = { i in I : teacher_prediction_i != ground_truth_i }
```

Evaluation terminology:

- **Test 1 = E1:** held-out teacher-correct examples;
- **Test 2 = E2:** teacher-error examples excluded from student training.

This separates two behaviors:

1. retention/generalization of useful teacher supervision;
2. behavior on examples for which the original teacher was wrong.

---

### 7. Verdict–Rationale Distillation

The principal compact backbone is **Gemma-3-1B-it**. **Gemma-2-2B-it** is evaluated as an additional backbone.

For each training example:

- the verification instruction and four evidence fields form the input prefix;
- the teacher JSON verdict–rationale response forms the target sequence.

WEAVE applies token-role weighting so that verdict prediction receives greater emphasis while rationale tokens continue to supervise evidence-grounded explanation generation.

```text
Label-token weight     = 5
Rationale-token weight = 1
```

Prompt and padding positions are masked from the loss.

---

### 8. QLoRA Adaptation

The compact verifier is adapted using **QLoRA**.

The quantized base model remains frozen while low-rank adapters are optimized.

Adapters cover:

- query projection;
- key projection;
- value projection;
- output projection;
- gate projection;
- up projection;
- down projection.

Paper training configuration:

| Item | Setting |
|---|---|
| Base quantization | 4-bit NF4; double quantization |
| Adapter rank / alpha / dropout | 16 / 32 / 0.05 |
| 1B adapter parameters | 13,045,760 |
| Epochs / microbatch | 5 / 1 example |
| Gradient accumulation | 4 steps |
| Optimizer / learning rate | AdamW / `2e-4` |
| Weight decay / warmup | 0.01 / 20 steps |
| Maximum gradient norm | 1.0 |
| Training sequence length | 1,024 tokens |
| Precision | Mixed precision; bf16 when supported |
| Checkpoint selection | Minimum validation loss |
| Training peak memory (1B) | 3.88 GB |
| Decoding | Greedy; maximum 256 new tokens |
| Student generation time | Approximately 12–15 s/example |
| Evaluation wall time | Approximately 4–5 h for over 1,300 examples |

No separate classifier head is required. The compact model directly generates the verdict and rationale.

---

## Datasets

The study uses material from **FakeSV** and **FakeTT**.

| Collection | Original | Usable | Real | Fake |
|---|---:|---:|---:|---:|
| FakeSV | 3,624 | 3,609 | 1,802 | 1,807 |
| FakeTT | 1,991 | 1,990 | 819 | 1,171 |
| **Total** | **5,615** | **5,599** | **2,621** | **2,978** |

The usable inventory contains labeled examples successfully processed by the large-model pipeline.

---

## Evaluation Protocol

### Teacher Development

The teacher development study compares:

1. an ExMRD-style direct large-model decision without SLR;
2. an initial WEAVE verification prompt;
3. the developed WEAVE verifier after DSPy-assisted instruction search and manual refinement.

| Configuration | Accuracy |
|---|---:|
| ExMRD-style direct decision, no SLR | 68.36% |
| WEAVE, initial verification prompt | 73.23% |
| **WEAVE, developed verification prompt** | **86.39%** |

---

### Student Evaluation Groups

| Group | Definition | Size |
|---|---|---:|
| Test 1 | Held-out teacher-correct examples | 507 |
| Test 2 | Teacher-error examples excluded from student training | 802 |
| Pooled diagnostic set | Test 1 + Test 2 | 1,309 |

Student classification is evaluated using:

- accuracy;
- macro precision;
- macro recall;
- macro F1.

Generated rationales are inspected separately for whether their stated evidence comparison is actually supported by the supplied record.

---

## Results

### WEAVE 1B

| Evaluation | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| **Test 1** | **88.17%** | **88.99%** | **88.09%** | **88.09%** |
| **Test 2** | **44.64%** | **36.30%** | **35.58%** | **35.91%** |
| **Pooled** | **61.50%** | **58.81%** | **58.73%** | **58.76%** |

Test 1 measures retention on held-out teacher-correct examples. Test 2 evaluates the harder teacher-error condition. The teacher-error responses are never supplied as student training targets, so successful predictions on Test 2 indicate behavior beyond simply reproducing the teacher's original labels.

### Student Backbone Comparison

| Backbone | Test 2 Accuracy | Pooled Accuracy |
|---|---:|---:|
| **Gemma-3-1B-it** | **44.64%** | **61.50%** |
| Gemma-2-2B-it | 27.81% | 48.20% |

The adapted Gemma-3-1B-it verifier outperforms Gemma-2-2B-it on both shared conditions.

---

## Computational Characteristics

WEAVE reallocates online computation.

| Measure | ExMRD + SLR | WEAVE |
|---|---:|---:|
| Remote model requests | 4 | **1** |
| Mean token use | 2,352.51 | 4,353.46 |
| Mean API time | 46.41 s | 128.28 s |

The single preparation request reduces remote invocations while using more tokens and API processing time. Once the evidence fields are available, the compact local student generates the final verdict and rationale.

---

## Claim–Evidence Inspection

A central design goal of WEAVE is that the generated reason should expose the comparison made by the model.

A useful rationale should identify:

1. the assertion being checked;
2. the relevant observation or event information;
3. the agreement or discrepancy between them.

A label can match the dataset annotation while its rationale is still weak or unsupported. Rationale quality is therefore not treated as equivalent to classification accuracy.

---

## Repository Structure

The repository preserves earlier experimental layouts for reproducibility.

```text
.
├── weave_bigdata26.pdf
├── README.md
├── preprocess/
├── CoT/
├── ExMRD/
├── LLM_5_Stage/
├── data/
│   └── ALL/
└── assets/
```

Existing code areas should not be interpreted as the final conceptual architecture of the paper.

---

## Data Organization

Raw data:

```text
data/ALL/
├── videos/
├── data.jsonl
└── label.jsonl
```

Existing preprocessed data:

```text
data/ALL/
├── ocr.jsonl
└── all.jsonl
```

Existing experimental outputs:

```text
data/ALL/
├── CoT/
│   └── gemma-4-31b-it/
├── Final_v1/
├── Final_v2/
├── Final_v3/
└── DSPy/
```

These directory names reflect the development history of the project and are retained for reproducibility.

---

## Running the Existing Codebase

Because the repository preserves the earlier experimental layout, the commands below describe the existing executable organization.

### 1. Prepare Data

Place videos under:

```text
data/ALL/videos/
```

and prepare:

```text
data/ALL/data.jsonl
data/ALL/label.jsonl
```

### 2. Existing Preprocessing Utilities

```bash
# Extract frames
python preprocess/1.Extract_Frames.py

# Extract audio
python preprocess/1.Video2Wav.py

# Optional / legacy redundant-frame filtering
python preprocess/2.Frames_Compress.py

# Generate speech transcripts
python preprocess/2.Wav2Transcript.py

# Optional / legacy OCR utility
python preprocess/3.Frames2OCR.py

# Merge available multimodal information
python preprocess/4.Data_Merge.py
```

The existing merged output is stored at:

```text
data/ALL/all.jsonl
```

### 3. Existing Evidence / Retrieval Script

```bash
python CoT/1.Retrieve.py
```

### 4. Existing Reasoning Script

```bash
python CoT/2.Reason.py
```

### 5. DSPy / MIPROv2 Prompt Development

```bash
python CoT/3.DSPy.py
```

### 6. Evaluation

```bash
python CoT/4.Accuracy.py
```

> The final paper additionally includes compact-student QLoRA adaptation and verdict–rationale generation. Historical script names are preserved rather than assigning an unverified training entry point to the final WEAVE stage.

---

## API Configuration

The Web-assisted evidence pipeline requires Gemini / Google GenAI API access.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY_1=YOUR_API_KEY
GEMINI_API_KEY_2=YOUR_API_KEY
```

Multiple API keys may be configured for parallel processing or key rotation.

> **Security:** Never commit `.env`, API keys, authentication cookies, access tokens, or other credentials to GitHub.

---

## Main Technologies

| Component | Technology |
|---|---|
| Speech recognition | Whisper Large V3 |
| Chronological visual description | Qwen3-VL-4B-Instruct |
| Evidence preparation / teacher | Gemma 4 31B IT via Gemini API |
| Web evidence | Google Search grounding |
| Prompt development | DSPy |
| Prompt optimizer | MIPROv2 |
| Principal compact verifier | Gemma-3-1B-it |
| Comparison compact backbone | Gemma-2-2B-it |
| Parameter-efficient adaptation | LoRA / QLoRA |
| Quantization | 4-bit NF4 |
| Video processing | OpenCV |
| Existing OCR utility | PaddleOCR |
| Existing frame-similarity utility | SSIM |
| Evaluation | Accuracy + macro Precision / Recall / F1 |

---

## Baselines and Historical Implementations

### ExMRD

The original ExMRD-based implementation is preserved under:

```text
ExMRD/
```

It is retained as the main historical baseline for comparison.

```text
Texture Refinement
        |
        v
Visual Refinement
        |
        v
Retrieval
        |
        v
Reasoning
        |
        v
Prediction
```

### Earlier LLM Five-Stage Pipeline

An intermediate implementation is retained under:

```text
LLM_5_Stage/
```

This directory represents an earlier stage of the project and is not the conceptual organization used to describe the final WEAVE framework.

### Earlier Refine–Retrieve–Reason Naming

Some scripts and the separate WebApp repository still use the earlier **Refine–Retrieve–Reason** naming. These names are retained to avoid unnecessary code and link breakage while the research method is documented under the final WEAVE formulation.

---

## Web Demo

An interactive web application is available in a separate repository:

### [Refine-Retrieve-Reason-FakeNews-WebApp](https://github.com/tyh1003/Refine-Retrieve-Reason-FakeNews-WebApp)

The WebApp repository keeps its original name and URL. It provides the interactive demonstration interface associated with this research project, while this repository focuses on the technical pipeline, experimental code, baselines, and compact-verifier study.

---

## Reproducibility Notes

- Some scripts contain experiment-specific paths and model configurations and may require adjustment on another machine.
- GPU acceleration is recommended for local multimodal processing and compact-model adaptation.
- Search-assisted evidence depends on the information available to the search-enabled model at verification time.
- The paper studies **retrospective verification**: Web retrieval may surface reporting or fact checks published after the original video.
- Search queries and source URLs should be retained as provenance metadata when reproducing the evidence-preparation stage.
- Model-response policies or service limitations can make a small number of dataset items unusable.
- Classification correctness and rationale support should be evaluated separately.
- The current codebase contains historical components from earlier project iterations; reproduction should distinguish final WEAVE components from preserved legacy utilities.

---

## Research Questions

### RQ1 — Teacher Development

How do the developed large-model configurations behave when preparing supervision?

### RQ2 — Compact Verifier Behavior

How does the compact verifier perform across teacher-correct and teacher-error groups, and across different student backbones?

### RQ3 — Rationale Evidence Relationships

How do generated rationales relate the video's claim to the supplied evidence, and when are those explanations actually supported by the evidence record?

---

## Web Demo

The existing WebApp URL is intentionally preserved:

### [Refine-Retrieve-Reason-FakeNews-WebApp](https://github.com/tyh1003/Refine-Retrieve-Reason-FakeNews-WebApp)

---

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{weave,
  title  = {From Web Evidence to Compact Verifiers: Distilling Verdicts and Rationales for Short-Video Misinformation Detection},
  author = {Chun-Yi Shih and Kai-Yun Hsiao and Yi-Hsien Tsai and Cheng-Te Li},
  note   = {WEAVE: Web Evidence-Assisted Video Examination}
}
```

Publication venue/year/DOI should be added when the final bibliographic record is available.

---

## Full Paper

For the complete methodology, experimental setup, results, and analysis:

**[Read the Full Paper](./weave_bigdata26.pdf)**
