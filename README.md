# WEAVE: Web Evidence-Assisted Video Examination

**From Web Evidence to Compact Verifiers: Distilling Verdicts and Rationales for Short-Video Misinformation Detection**

Research code for **WEAVE (Web Evidence-Assisted Video Examination)**, a framework for evidence-grounded short-video misinformation detection that separates multimodal evidence preparation from compact verdict–rationale generation.

WEAVE reconstructs claims from short-video content, organizes the available information into four source-aware evidence fields, uses a large teacher to produce offline verdict–rationale supervision, and adapts a compact language model to generate the final decision and explanation.

> **Interactive demo**  
> The original WebApp repository is retained here:  
> [WEAVE-WebApp](https://github.com/tyh1003/WEAVE-WebApp)

---

## Paper

**Title:** *From Web Evidence to Compact Verifiers: Distilling Verdicts and Rationales for Short-Video Misinformation Detection*

**Authors:** Chun-Yi Shih, Kai-Yun Hsiao, Yi-Hsien Tsai, Cheng-Te Li  
**Affiliation:** Department of Computer Science and Information Engineering, National Cheng Kung University, Taiwan

### Keywords

`short-video misinformation` · `Web evidence` · `multimodal verification` · `knowledge distillation` · `compact language models` · `verdict-rationale generation`

---

## Overview

A short video can contain genuine footage while still communicating a misleading claim. The mismatch may come from a changed caption, an incorrect date or location, reused footage, an unrelated narrative, or a false statement shown visually on screen.

This makes short-video misinformation detection more than a conventional video-classification problem. A verifier must answer three questions:

1. **What claim is the video actually conveying?**
2. **What does the video itself show or say?**
3. **Does reliable background information support or contradict that claim?**

WEAVE addresses this problem by separating the system into two computational roles:

- a **large-model evidence-preparation and supervision stage**, which reconstructs claims, organizes multimodal observations, retrieves Web-assisted context, and produces offline teacher responses;
- a **compact generative verifier**, which learns to map the prepared evidence to a structured verdict and rationale.

The final compact verifier does not receive raw video at inference time. Instead, it operates on an evidence record that explicitly separates the textual claim, video account, model background, and Web-assisted context.

---

## Main Contributions

### 1. Claim-centered evidence interface

WEAVE represents each video using four text fields with explicit information roles:

- `Rc` — reconstructed textual claim;
- `Rv` — video account containing visual claims and event observations;
- `K_int` — model-generated internal/background knowledge;
- `K_ext` — Web-assisted external context.

This representation is designed to preserve claims regardless of whether they appear in a title, speech, or visually displayed text.

### 2. Joint verdict–rationale supervision

A large teacher produces structured responses containing both:

- a binary misinformation verdict; and
- a concise natural-language rationale.

The student is trained with separately normalized label and rationale losses so that the short verdict is not overwhelmed by the larger number of rationale tokens.

### 3. Compact evidence-conditioned verification

The compact verifier learns the final evidence comparison rather than merely reproducing an intermediate representation. At inference time, it generates both the verdict and the explanation from the prepared evidence record without teacher involvement.

### 4. Teacher-conditioned evaluation

Student behavior is analyzed separately on:

- **Test 1:** held-out examples originally predicted correctly by the teacher;
- **Test 2:** examples originally misclassified by the teacher and excluded from student training.

This distinguishes retention of useful teacher behavior from student behavior on teacher-error cases.

### 5. Inspectable generated rationales

A correct label does not automatically imply that its explanation is supported by the evidence. WEAVE therefore treats rationale inspection as a separate analysis target and examines whether the generated explanation states an identifiable claim–evidence agreement or discrepancy.

---

## WEAVE Architecture

```text
                             Short Video
                       + Title / Description
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
      Audio Transcription                   Visual Processing
     Whisper Large V3               ┌──────────────┴──────────────┐
             │                      │                             │
             │               Representative Frames      Chronological Description
             │                    up to 16 frames          Qwen3-VL-4B-Instruct
             │                      │                             │
             └──────────────────────┴──────────────┬──────────────┘
                                                  │
                                                  ▼
                               Web-Assisted Evidence Preparation
                              Gemma 4 31B IT + Google Search
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         │                        │                        │
                         ▼                        ▼                        ▼
                        Rc                       Rv                  K_int / K_ext
                 Textual Claim             Video Account              Background
                         └────────────────────────┬────────────────────────┘
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       │                                                     │
                       │ OFFLINE TRAINING                                    │ INFERENCE
                       ▼                                                     ▼
              Large Teacher Response                              Compact Student Verifier
              verdict + rationale                                  evidence comparison
                       │                                                     │
                       ▼                                                     ▼
              Teacher supervision                                  {pred_label, reason}
                       │
                       ▼
          QLoRA adaptation of compact student
          with label/rationale segment losses
```

The key architectural distinction is that **evidence preparation** and **final response generation** are separate tasks. Large-model computation is used upstream to prepare structured evidence, while the compact student learns the final claim–evidence comparison.

---

## Problem Formulation

For video example `i`, let

- `b_i` denote the video frames;
- `u_i` denote the audio track;
- `t_i` denote the title or description;
- `y_i ∈ {0,1}` denote the dataset label, where `0 = real` and `1 = fake/misleading`.

Given Web information `W_i`, the evidence-preparation function constructs

```text
x_i = [Rc_i, Rv_i, K_int,i, K_ext,i]
```

where:

| Field | Meaning | Primary role |
|---|---|---|
| `Rc` | Reconstructed textual claim | Recover assertions expressed through title and speech |
| `Rv` | Video account | Preserve visual claims, visible assertions, and event observations |
| `K_int` | Internal/model background | Supply related entity and event knowledge |
| `K_ext` | Web-assisted context | Compare the claim against external accounts |

The compact verifier receives these four fields plus a verification instruction and generates:

```json
{
  "pred_label": 0,
  "reason": "A concise evidence-grounded explanation of the decisive agreement or discrepancy."
}
```

---

## Method

### 1. Multimodal Content Representation

WEAVE accepts a short video together with its title or description.

### 1.1 Speech transcription

Audio is transcribed with **Whisper Large V3**.

If the video contains no usable audio, or transcription fails, the transcript is treated as an empty string and the remaining modalities continue through the pipeline.

### 1.2 Representative-frame view

A visual path supplies up to **16 representative images** distributed over the duration of the video.

For a video with duration `d > 0`, target frame times are distributed from the beginning to the end of the clip. Failed frame reads are skipped, and the final target uses the last decodable frame near the end of the video.

The purpose of this view is to preserve fine visual information, especially short-lived on-screen assertions that may not appear in the audio transcript.

### 1.3 Chronological visual description

A second visual path uses **Qwen3-VL-4B-Instruct** to construct a temporal description of the video.

Paper configuration:

| Setting | Value |
|---|---|
| Visual model | Qwen3-VL-4B-Instruct |
| Sampling | 5 FPS |
| Maximum sampled frames | 120 |
| Image resolution | Dynamic |
| Quantization | 4-bit NF4 |
| Example hardware | RTX 5070 Laptop GPU, 8 GB |

The visual-description prompt used in the paper is:

> Objectively describe the visual content of the video in temporal order. Respond in Traditional Chinese.

The representative images and chronological description are complementary: images preserve local details, while the generated description summarizes event order.

---

### 2. Web-Assisted Evidence Construction

The paper uses **Gemma 4 31B IT** through the Gemini API with **temperature 0** and **Google Search enabled**.

A single external-model preparation request constructs the four evidence fields.

### `Rc`: reconstructed textual claim

Constructed from:

- the title or description;
- the speech transcript.

Its role is to recover the explicit textual assertion rather than provide a generic summary.

### `Rv`: video account

Constructed from:

- `Rc`;
- the chronological visual description;
- the ordered representative frames.

`Rv` should preserve:

- visible claims;
- people and entities;
- actions;
- locations;
- dates or temporal cues;
- event order;
- observations needed for fact checking.

### `K_int`: internal background

Contains background supplied from the model's parametric knowledge.

This field is kept separate from search-derived evidence so that the source role of each piece of information remains explicit.

### `K_ext`: Web-assisted context

Contains information obtained through search-assisted verification.

Searches should target the specific event and use relevant combinations of:

- entity names;
- actions;
- locations;
- dates;
- event-specific terminology.

The paper favors authoritative sources and stores deduplicated search queries and source links as provenance metadata outside the four-field student input.

---

### 3. Verification Contract

The verifier is instructed to compare the assertion carried by the video with the supplied evidence rather than merely detect familiar entities.

The comparison includes checks for:

- reused footage with changed context;
- reassigned locations or settings;
- incorrect entities;
- incorrect dates or other precise details;
- attachment of an unrelated narrative;
- physical implausibility;
- contradictions between textual claims and visible events.

Important evidence rules include:

- use both internal and external background when making the comparison;
- do not infer falsity only because search results are missing;
- do not invent absent dates or details and then use them as contradictions;
- explain the decisive agreement or discrepancy rather than simply restating the predicted label.

### Output contract

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

### 4. Teacher Response Generation

The large-model teacher receives the four-field evidence record and the finalized verification instruction.

For each example, the teacher generates:

```text
(predicted label, rationale)
```

These responses are used as offline supervision for the compact verifier.

The student therefore learns from **teacher-generated structured responses**, rather than from teacher logits.

---

### 5. DSPy / MIPROv2 Prompt Development

**DSPy** with **MIPROv2** is used during development of the verification instruction and demonstrations.

It is used for the teacher reasoning stage **after the evidence fields have been prepared**.

The development objective places most of its weight on prediction quality while retaining a smaller criterion for rationale completeness:

```text
Development objective
│
├── 95%  prediction quality
│
└──  5%  rationale completeness
```

The paper's development score uses:

- `0.95 × (1 - squared label/probability error)`;
- an additional `0.05` when the stripped reason contains at least 50 characters.

DSPy and MIPROv2 are **development-time tools**. They are not executed by the compact student during inference.

---

### 6. Teacher-Correct Supervision Selection

Let `I` denote examples with aligned evidence records, ground-truth labels, and teacher responses.

The teacher-correct pool is

```text
C = { i ∈ I : teacher_prediction_i = ground_truth_i }
```

`C` is split with class stratification into:

- training set `D_tr`;
- validation set `D_val`;
- held-out teacher-correct set `E1`.

Teacher-error examples form:

```text
E2 = { i ∈ I : teacher_prediction_i != ground_truth_i }
```

The evaluation terminology is:

- **Test 1 = E1:** held-out teacher-correct examples;
- **Test 2 = E2:** teacher-error examples excluded from student training.

This protocol makes two student behaviors separately observable:

1. retention/generalization on useful teacher supervision;
2. behavior on examples for which the original teacher was wrong.

---

### 7. Verdict–Rationale Distillation

The principal compact backbone is **Gemma-3-1B-it**. **Gemma-2-2B-it** is evaluated as an additional backbone.

For training example `i`:

- the verification instruction and the four evidence fields form the input prefix `h_i`;
- the teacher's JSON label–reason response forms the target sequence `z_i`.

The supervised target positions are divided into:

- `T_i^y`: label-token positions;
- `T_i^e`: rationale-token positions.

The segment loss is normalized separately:

```math
L_i^k = -\frac{1}{|T_i^k|}\sum_{t \in T_i^k}
\log p_\theta(z_{it}\mid h_i,z_{i,<t}),
\qquad k\in\{y,e\}
```

and the total training objective is

```math
L = \frac{1}{|D_{tr}|}\sum_{i\in D_{tr}}
\left(\lambda_y L_i^y + \lambda_e L_i^e\right)
```

with

```text
lambda_y = 5
lambda_e = 1
```

### Why separate normalization?

The rationale usually contains many more tokens than the verdict. If ordinary token averaging were used without controlling the two segments, rationale length could dominate the objective simply because it contains more tokens.

WEAVE therefore computes separate segment means and assigns an explicit larger coefficient to the verdict.

This gives the system two simultaneous learning targets:

- prioritize correct label generation;
- preserve supervision for evidence-grounded explanation generation.

---

### 8. QLoRA Adaptation

The compact verifier is adapted with **QLoRA**.

The quantized base model remains frozen while low-rank adapters are optimized.

Adapters cover:

- query projection;
- key projection;
- value projection;
- output projection;
- gate projection;
- up projection;
- down projection.

### Paper training configuration

| Item | Setting |
|---|---|
| Principal backbone | Gemma-3-1B-it |
| Comparison backbone | Gemma-2-2B-it |
| Base quantization | 4-bit NF4 + double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| 1B adapter parameters | 13,045,760 |
| Epochs | 5 |
| Microbatch | 1 example |
| Gradient accumulation | 4 steps |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Weight decay | 0.01 |
| Warmup | 20 steps |
| Maximum gradient norm | 1.0 |
| Training sequence length | 1,024 tokens |
| Precision | mixed precision; bf16 when supported |
| Checkpoint selection | minimum validation loss |
| 1B peak training memory | 3.88 GB |

---

### 9. Compact Inference

At inference time:

1. the large preparation model constructs `Rc`, `Rv`, `K_int`, and `K_ext`;
2. the compact student receives the verification instruction and four evidence fields;
3. the backbone chat template formats the input;
4. greedy autoregressive generation produces the response;
5. only newly generated tokens are decoded;
6. the JSON parser extracts `pred_label` and `reason`.

Paper decoding configuration:

| Item | Setting |
|---|---|
| Decoding | Greedy |
| Maximum new tokens | 256 |
| Local student generation time | approximately 12–15 s/example |
| Evaluation wall time | approximately 4–5 h for more than 1,300 examples |

No separate classifier head is required: the student directly generates the verdict and rationale.

---

## Datasets

The study combines material from **FakeSV** and **FakeTT**.

| Dataset | Original | Usable | Real | Fake |
|---|---:|---:|---:|---:|
| FakeSV | 3,624 | 3,609 | 1,802 | 1,807 |
| FakeTT | 1,991 | 1,990 | 819 | 1,171 |
| **Total** | **5,615** | **5,599** | **2,621** | **2,978** |

Publication ranges used in the study:

- **FakeSV:** October 2017 – February 2022
- **FakeTT:** May 2019 – March 2024

The usable set contains labeled examples that could be successfully processed by the large-model pipeline. Items unavailable because of model-response policies or other processing limitations are excluded.

---

## Evaluation Protocol

### Teacher development

The teacher development study compares:

1. an ExMRD-style direct large-model decision without SLR;
2. an initial WEAVE verification prompt;
3. the developed WEAVE verifier after DSPy-assisted instruction search and manual refinement.

### Student evaluation groups

| Group | Definition | Size |
|---|---|---:|
| Test 1 | Held-out teacher-correct examples | 507 |
| Test 2 | Teacher-error examples excluded from student training | 802 |
| Pooled diagnostic set | Test 1 ∪ Test 2 | 1,309 |

### Metrics

Student classification is evaluated with:

- accuracy;
- macro precision;
- macro recall;
- macro F1.

Macro metrics are used so that both classes contribute equally to the reported score.

Generated rationales are also inspected separately for whether their stated evidence comparison is actually supported by the supplied record.

---

## Results

### Teacher Development Configurations

| Configuration | Accuracy |
|---|---:|
| ExMRD-style direct decision, no SLR | 68.36% |
| WEAVE, initial verification prompt | 73.23% |
| **WEAVE, developed verification prompt** | **86.39%** |

These are configuration-level teacher-development measurements. They should not be interpreted as the compact student's Test 1/Test 2 results.

The developed verification instruction places greater emphasis on event-level comparisons: matching a person alone is not sufficient when the claim concerns a different location, quotation, date, or event.

---

### Compact Student Behavior

The paper reports the student results by teacher-conditioned evaluation group rather than treating the pooled set as the only result.

Observed behavior includes:

- strong retention on held-out teacher-correct examples in Test 1;
- correct predictions on a subset of teacher-error examples in Test 2;
- lower scores on Test 2 than Test 1, showing that teacher-error examples are the harder condition;
- a stronger recorded decision profile for the adapted **Gemma-3-1B-it** configuration than the evaluated **Gemma-2-2B-it** configuration in the teacher-error and pooled conditions.

The pooled accuracy depends on both conditional accuracies and the mixture of Test 1/Test 2 examples. For this reason, the paper reports the conditional groups separately.

---

### Computational Characteristics

WEAVE separates external-model computation from local student generation.

### External preparation

The Web-assisted preparation stage performs content reconstruction, model background generation, and search-assisted context construction in a single external-model request.

Compared with the multi-call ExMRD-style pipeline, WEAVE reduces the number of remote model invocations, while the preparation request itself can use more tokens and API processing time.

### Local verifier

The compact student performs the final evidence-to-response mapping locally using its frozen quantized backbone and learned adapters.

This stage is a generative verifier rather than a single forward-pass classification head because it autoregressively generates both the verdict and its rationale.

---

## Claim–Evidence Inspection

A central design goal of WEAVE is that the generated reason should expose the comparison made by the model.

A useful rationale should identify:

1. the assertion being checked;
2. the relevant observation or event information;
3. the agreement or discrepancy between them.

A label can match the dataset annotation while its rationale is still weak or unsupported. Rationale quality is therefore not treated as equivalent to classification accuracy.

### Example: visually expressed claim

A screen recording may display a shutdown announcement while its transcript contains only a generic closing phrase. In this case:

- `Rc` alone may contain little useful information;
- `Rv` can recover the visible assertion from the video frames;
- `K_ext` can provide event-specific external context;
- the compact verifier can then state the decisive comparison in its generated reason.

This illustrates why WEAVE preserves claim-bearing visual content instead of relying only on speech or titles.

---

## Repository Structure

The repository keeps much of the original experimental directory layout so that earlier scripts and baselines remain reproducible.

```text
WEAVE-ShortVideo-Misinformation-Detection/
│
├── preprocess/
│   ├── 1.Extract_Frames.py
│   ├── 1.Video2Wav.py
│   ├── 2.Frames_Compress.py
│   ├── 2.Wav2Transcript.py
│   ├── 3.Frames2OCR.py
│   └── 4.Data_Merge.py
│
├── CoT/
│   ├── 1.Retrieve.py
│   ├── 2.Reason.py
│   ├── 3.DSPy.py
│   ├── 4.Accuracy.py
│   ├── 5.Error.py
│   └── Refine_compare.py
│
├── ExMRD/
│   ├── 1.texture_refine.py
│   ├── 2.visual_refine.py
│   ├── 3.retrieve.py
│   ├── 4.reason.py
│   ├── 5.predict.py
│   ├── 6.evaluate.py
│   ├── 7.baseline.py
│   ├── extract_frame_original.py
│   ├── frames_to_quad_4_original.py
│   └── ocr_original.py
│
├── LLM_5_Stage/
│   └── ...
│
├── data/
│   └── ALL/
│
├── assets/
│   └── system_pipeline.png
│
└── README.md
```

---

### Codebase / Paper Alignment

The research direction evolved from the earlier **Refine–Retrieve–Reason** pipeline into the paper's final **WEAVE** formulation. The codebase intentionally retains several original file and directory names.

This means that **directory names should not be interpreted as the final conceptual architecture**.

| Existing code area | Role in the repository | Relation to final WEAVE paper |
|---|---|---|
| `preprocess/` | Video/audio/OCR preprocessing utilities | Contains reusable and legacy preprocessing code; the paper emphasizes Whisper, representative frames, and chronological VLM description |
| `CoT/` | Retrieval/reasoning/prompt-development experiments | Closest existing implementation area for evidence preparation and teacher prompt development |
| `ExMRD/` | ExMRD reproduction / baseline | Baseline and comparison implementation |
| `LLM_5_Stage/` | Earlier experimental pipeline | Historical intermediate implementation |
| `data/ALL/` | Data, intermediate outputs, predictions, experiments | Shared experimental data organization |
| `assets/` | Repository figures | May contain figures from earlier project iterations |

### Important note on legacy preprocessing

Scripts such as SSIM-based frame compression and PaddleOCR remain useful experimental utilities and are preserved in the repository. However, they should not be presented as defining components of the final paper architecture unless they are explicitly enabled in a particular reproduction configuration.

The final paper methodology centers on:

```text
Whisper transcript
+ representative frames
+ Qwen3-VL chronological description
        │
        ▼
Rc / Rv / K_int / K_ext evidence construction
        │
        ▼
Teacher verdict-rationale supervision
        │
        ▼
QLoRA-adapted compact verifier
```

---

## Data Organization

The existing repository stores dataset metadata and experiment outputs under:

```text
data/ALL/
```

### Raw data

```text
data/ALL/
├── videos/       # source short-form videos
├── data.jsonl    # video metadata / content records
└── label.jsonl   # ground-truth labels
```

### Existing preprocessed data

```text
data/ALL/
├── ocr.jsonl
└── all.jsonl
```

### Existing experimental outputs

```text
data/ALL/
│
├── CoT/
│   └── gemma-4-31b-it/
│
├── Final_v1/
│
├── Final_v2/
│
├── Final_v3/
│
└── DSPy/
```

These directory names reflect the development history of the project and are retained for reproducibility.

---

## Running the Existing Codebase

Because the repository preserves the earlier experimental layout, the commands below describe the **existing executable organization**. They should be interpreted together with the paper-alignment notes above.

### 1. Prepare data

Place videos under:

```text
data/ALL/videos/
```

and prepare:

```text
data/ALL/data.jsonl
data/ALL/label.jsonl
```

### 2. Existing preprocessing utilities

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

### 3. Existing evidence/retrieval script

```bash
python CoT/1.Retrieve.py
```

### 4. Existing reasoning script

```bash
python CoT/2.Reason.py
```

### 5. DSPy / MIPROv2 prompt development

```bash
python CoT/3.DSPy.py
```

### 6. Evaluation

```bash
python CoT/4.Accuracy.py
```

> The final paper additionally includes compact-student QLoRA adaptation and verdict–rationale generation. The exact training entry point should follow the experiment-specific training code in the repository; this README intentionally does not assign an unverified script path to that stage.

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
        │
        ▼
Visual Refinement
        │
        ▼
Retrieval
        │
        ▼
Reasoning
        │
        ▼
Prediction
```

The WEAVE paper uses an ExMRD-style direct large-model decision without SLR as one teacher-development reference configuration and separately discusses the multi-stage ExMRD computation profile.

### Earlier LLM five-stage pipeline

An intermediate implementation is retained under:

```text
LLM_5_Stage/
```

This directory represents an earlier stage of the project and is not the conceptual organization used to describe the final WEAVE framework.

### Earlier Refine–Retrieve–Reason naming

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

The paper is organized around three research questions:

### RQ1 — Teacher development

How do the developed large-model configurations behave when preparing supervision?

### RQ2 — Compact verifier behavior

How does the compact verifier perform across teacher-correct and teacher-error groups, and across different student backbones?

### RQ3 — Rationale evidence relationships

Which claim–evidence relationships appear in the generated rationales, and are those relationships supported by the supplied evidence?

---

## Design Principles

The final WEAVE formulation follows several practical principles:

1. **Preserve the claim wherever it appears.**  
   Important assertions may appear in speech, captions, titles, or briefly visible screen text.

2. **Compare events, not only recognizable entities.**  
   Matching a person or topic is insufficient when the claim concerns a different action, date, location, quotation, or event.

3. **Keep evidence roles explicit.**  
   Model recollection and Web-derived information are separated into `K_int` and `K_ext`.

4. **Make the verdict's training weight explicit.**  
   Separate label/rationale normalization prevents long explanations from dominating the learning objective only because they contain more tokens.

5. **Evaluate student behavior conditionally.**  
   Teacher-correct and teacher-error groups reveal different aspects of knowledge transfer.

6. **Inspect what the rationale actually claims.**  
   A correct binary prediction does not guarantee a supported explanation.

---

## Limitations and Future Directions

The paper identifies several directions for further development:

- claim-focused evidence selection under a fixed context budget;
- passage-linked rationale supervision;
- source-separated evaluation;
- event-separated evaluation;
- temporal evidence policies that distinguish retrospective verification from time-restricted verification;
- explicit evidence-insufficient outcomes when appropriate labels and evaluation protocols are available.

These directions aim to make the compact verifier's generated explanation more directly traceable to the evidence supporting it.

---

## Citation

If you use this repository in academic work, please cite the paper:

> **From Web Evidence to Compact Verifiers: Distilling Verdicts and Rationales for Short-Video Misinformation Detection**  
> Chun-Yi Shih, Kai-Yun Hsiao, Yi-Hsien Tsai, Cheng-Te Li.  
> Department of Computer Science and Information Engineering, National Cheng Kung University, Taiwan.

Publication-specific BibTeX information can be added here once the final venue / bibliographic record is available.

---

## Notes

- The current repository name recommended for the paper-aligned codebase is **`WEAVE-ShortVideo-Misinformation-Detection`**.
- The codebase retains older directory names to preserve experimental history and minimize unnecessary implementation changes.
- The final conceptual method should be referred to as **WEAVE (Web Evidence-Assisted Video Examination)** rather than Refine–Retrieve–Reason.
- The original interactive WebApp GitHub link is intentionally preserved.
