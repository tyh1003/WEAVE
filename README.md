# Refine-Retrieve-Reason-FakeNews

**An LLM-powered Refine–Retrieve–Reason framework for multimodal fake news detection in short-form videos.**

This repository contains the technical implementation of our **Refine–Retrieve–Reason (R³)** framework, including multimodal preprocessing, evidence retrieval, LLM-based reasoning, DSPy prompt optimization, and evaluation.

> **Looking for the interactive demo?**
> See [Refine-Retrieve-Reason-FakeNews-WebApp](https://github.com/tyh1003/Refine-Retrieve-Reason-FakeNews-WebApp).

---

## Overview

Short-form videos combine visual content, on-screen text, speech, and contextual claims, making misinformation detection inherently multimodal.

Our framework processes these heterogeneous signals and performs evidence-grounded fake news detection through three main stages:

* **Refine** — extract and refine multimodal information from video frames, OCR, and speech transcripts.
* **Retrieve** — identify verifiable claims and retrieve relevant external evidence through web search.
* **Reason** — combine multimodal information with retrieved evidence to estimate the probability of misinformation.

We further integrate **DSPy with MIPROv2** to automatically optimize the reasoning pipeline.

The final system achieves **88.56% accuracy** and an **F1 score of 0.8852**, compared with **68.36% accuracy** and **0.6068 F1** for the original ExMRD baseline.

---

## System Architecture

<img width="1920" height="1080" alt="Step1_ 影像萃取與拼貼 - 1" src="https://github.com/user-attachments/assets/ce454f97-16e7-44b0-994a-2a44fff076e3" />

The framework follows a **Refine → Retrieve → Reason** pipeline:

```text
                         Short Video
                              │
               ┌──────────────┴──────────────┐
               │                             │
          Video Frames                     Audio
               │                             │
        Frame Filtering              Speech Transcription
               │                             │
              OCR                            │
               └──────────────┬──────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     REFINE      │
                    │   Multimodal    │
                    │   Information   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    RETRIEVE     │
                    │ Claim & Evidence│
                    │    Retrieval    │
                    └────────┬────────┘
                             │
                             ▼
                       Web Evidence
                             │
                             ▼
                    ┌─────────────────┐
                    │     REASON      │
                    │  LLM Reasoning  │
                    └────────┬────────┘
                             │
                             ▼
                  Misinformation Probability
```

---

## Refine

The **Refine** stage converts raw short-form videos into structured multimodal information.

The preprocessing pipeline is implemented in:

```text
preprocess/
```

### Frame Extraction

```text
preprocess/1.Extract_Frames.py
```

Representative frames are extracted from each input video for subsequent visual and OCR processing.

### Audio Extraction

```text
preprocess/1.Video2Wav.py
```

Audio tracks are extracted from the input videos for speech recognition.

### Frame Filtering

```text
preprocess/2.Frames_Compress.py
```

Short-form videos frequently contain consecutive frames with nearly identical visual information.

Redundant frames are filtered using the **Structural Similarity Index Measure (SSIM)**, reducing unnecessary downstream processing.

### Speech Transcription

```text
preprocess/2.Wav2Transcript.py
```

Speech is transcribed using **Whisper Large V3**:

```text
openai/whisper-large-v3
```

The transcripts provide the spoken context of each video.

### Optical Character Recognition

```text
preprocess/3.Frames2OCR.py
```

On-screen text is extracted from representative video frames using **PaddleOCR**.

The OCR pipeline includes:

* confidence filtering
* text normalization
* duplicate-text removal

### Multimodal Data Integration

```text
preprocess/4.Data_Merge.py
```

The extracted information is merged into a unified representation:

```text
Video Content
      +
   OCR Text
      +
Speech Transcript
      │
      ▼
Multimodal Representation
```

The resulting data is used as the input to the retrieval stage.

---

## Retrieve

The **Retrieve** stage identifies information that requires verification and retrieves relevant external evidence.

Main implementation:

```text
CoT/1.Retrieve.py
```

The retrieval pipeline receives the refined multimodal information:

```text
Video Content
OCR Text
Speech Transcript
      │
      ▼
Claim Identification
      │
      ▼
Query Generation
      │
      ▼
Google Search
      │
      ▼
External Evidence
```

An LLM analyzes the multimodal content to identify claims that require verification.

Relevant external information is then retrieved through **Google Search grounding via the Gemini API** and provided to the reasoning stage as background knowledge.

---

## Reason

The **Reason** stage combines the refined video information with retrieved external evidence to perform the final misinformation assessment.

Main implementation:

```text
CoT/2.Reason.py
```

The model estimates:

```text
P(misinformation | video content, retrieved evidence)
```

The prediction is represented as a probability between `0` and `1`:

| Score | Interpretation                          |
| ----: | --------------------------------------- |
| `0.0` | Highly likely to be true                |
| `0.5` | Uncertain                               |
| `1.0` | Highly likely to contain misinformation |

Along with the probability score, the system generates a natural-language explanation supporting its prediction.

---

## DSPy Optimization

To improve the reasoning stage, we use **DSPy** to automatically optimize the LLM pipeline instead of relying solely on manually designed prompts.

Implementation:

```text
CoT/3.DSPy.py
```

The DSPy reasoning program takes:

```text
Rc                    Textual core information
Rv                    Visual core information
background_knowledge  Retrieved evidence
```

and produces:

```text
pred_label             Misinformation probability
reason                 Reasoning explanation
```

The reasoning module is implemented using **DSPy ChainOfThought** and optimized using **MIPROv2**.

Our optimization objective is composed of:

```text
Optimization Score
│
├── 95%  Prediction Quality
│
└──  5%  Reasoning Completeness
```

DSPy training and optimization artifacts are stored under:

```text
data/ALL/DSPy/
```


---

## Experimental Results

### Model Performance

We evaluate the proposed framework against the original ExMRD baseline and different variants of our approach.

| Method              |   Accuracy |     Recall |  Precision |         F1 |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Baseline (ExMRD)    |     68.36% |     0.4599 |     0.8917 |     0.6068 |
| Our (LLM)           |     73.23% |     0.6013 |     0.8550 |     0.7060 |
| **Our (DSPy)**      | **87.51%** |     0.8270 | **0.9302** |     0.8756 |
| Baseline (with SLR) |     85.97% |     0.8532 |     0.8552 |     0.8580 |
| **Our (SLLM)**      | **88.56%** | **0.8851** |     0.8900 | **0.8852** |

Our final configuration achieves the highest overall:

* **Accuracy: 88.56%**
* **Recall: 0.8851**
* **F1 Score: 0.8852**

Compared with the original ExMRD baseline, accuracy increases from:

```text
68.36%  ───────────────►  88.56%
 ExMRD       +20.20 pp      Ours
```

DSPy optimization also substantially improves the LLM-based pipeline:

```text
73.23%  ───────────────►  87.51%
  LLM        +14.28 pp      DSPy
```


---

### Inference Cost

We also compare the inference cost of the original ExMRD pipeline with our proposed approach.

| Method           |  Avg. Tokens | Avg. Time | LLM Requests |
| ---------------- | -----------: | --------: | -----------: |
| Baseline (ExMRD) | **2,352.51** | **46.41** |            4 |
| Our Method       |     4,107.09 |     49.34 |        **1** |

Our method uses more tokens but significantly reduces the number of separate LLM requests.

```text
ExMRD

LLM ──► LLM ──► LLM ──► LLM
              4 requests


Refine–Retrieve–Reason

              LLM
           1 request
```

The number of LLM requests is reduced from **4 to 1**, while maintaining a comparable overall execution time.


---

## Evaluation

Evaluation is implemented in:

```text
CoT/4.Accuracy.py
```

The system is evaluated using both classification and probability-based metrics.

| Metric          | Description                                        |
| --------------- | -------------------------------------------------- |
| **Accuracy**    | Overall proportion of correct predictions          |
| **Recall**      | Ability to detect misinformation samples           |
| **Precision**   | Reliability of misinformation predictions          |
| **F1 Score**    | Balance between precision and recall               |
| **Brier Score** | Quality and calibration of probability predictions |
| **Log Loss**    | Confidence-aware probabilistic prediction error    |

---

## Repository Structure

```text
Refine-Retrieve-Reason-FakeNews/
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

### Directory Description

| Directory      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `preprocess/`  | Multimodal preprocessing pipeline                                           |
| `CoT/`         | Main Refine–Retrieve–Reason implementation and DSPy optimization            |
| `ExMRD/`       | Original ExMRD implementation used as the baseline                          |
| `LLM_5_Stage/` | Intermediate five-stage LLM implementation                                  |
| `data/`        | Dataset metadata, intermediate outputs, predictions, and experiment results |
| `assets/`      | Figures and images used in this README                                      |

---

## Data Organization

The dataset and experimental results are organized under:

```text
data/ALL/
```

### Raw Data

```text
data/ALL/
├── videos/       # Original short-form videos
├── data.jsonl    # Video data
└── label.jsonl   # Ground-truth labels
```

### Preprocessed Data

```text
data/ALL/
├── ocr.jsonl     # OCR results
└── all.jsonl     # Integrated multimodal data
```

### Experimental Results

```text
data/ALL/
│
├── CoT/
│   └── gemma-4-31b-it/
│       └── Original / baseline results
│
├── Final_v1/
│   └── Results before DSPy optimization
│
├── Final_v2/
│   └── DSPy-optimized results
│       Model: gemini-3.1-flash-lite
│
├── Final_v3/
│   └── DSPy-optimized results
│       Model: gemma-4-31b-it
│
└── DSPy/
    └── DSPy training and optimization artifacts
```

---

## Running the Pipeline

### 1. Prepare the Data

Place the source videos under:

```text
data/ALL/videos/
```

and prepare:

```text
data/ALL/data.jsonl
data/ALL/label.jsonl
```

### 2. Run Preprocessing

```bash
# Extract frames
python preprocess/1.Extract_Frames.py

# Extract audio
python preprocess/1.Video2Wav.py

# Remove redundant frames
python preprocess/2.Frames_Compress.py

# Generate speech transcripts
python preprocess/2.Wav2Transcript.py

# Run OCR
python preprocess/3.Frames2OCR.py

# Merge multimodal information
python preprocess/4.Data_Merge.py
```

The resulting multimodal data is stored in:

```text
data/ALL/all.jsonl
```

### 3. Retrieve Evidence

```bash
python CoT/1.Retrieve.py
```

### 4. Perform Reasoning

```bash
python CoT/2.Reason.py
```

### 5. Run DSPy Optimization

```bash
python CoT/3.DSPy.py
```

### 6. Evaluate

```bash
python CoT/4.Accuracy.py
```

---

## API Configuration

The retrieval and reasoning pipeline requires access to the **Gemini API**.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY_1=YOUR_API_KEY
GEMINI_API_KEY_2=YOUR_API_KEY
```

Multiple API keys can be configured for parallel requests and key rotation.

> **Important:** Never commit `.env`, API keys, cookies, or other credentials to GitHub.

---

## Main Technologies

| Component             | Technology              |
| --------------------- | ----------------------- |
| Large Language Models | Gemma / Gemini          |
| LLM API               | Google GenAI            |
| Evidence Retrieval    | Google Search Grounding |
| Prompt Optimization   | DSPy                    |
| DSPy Optimizer        | MIPROv2                 |
| LLM Reasoning         | Chain-of-Thought        |
| Speech Recognition    | Whisper Large V3        |
| OCR                   | PaddleOCR               |
| Frame Similarity      | SSIM                    |
| Video Processing      | OpenCV                  |
| Evaluation            | scikit-learn            |

---

## Baseline & Experimental Implementations

### ExMRD

The original ExMRD-based implementation is preserved under:

```text
ExMRD/
```

and serves as the primary baseline for comparison.

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

### LLM Five-Stage Pipeline

An intermediate experimental version is preserved under:

```text
LLM_5_Stage/
```

This implementation represents an earlier stage in the development of the final Refine–Retrieve–Reason framework.

---

## Web Demo

An interactive web application is available in a separate repository:

### [Refine-Retrieve-Reason-FakeNews-WebApp](https://github.com/tyh1003/Refine-Retrieve-Reason-FakeNews-WebApp)

> An interactive web application showcasing the Refine–Retrieve–Reason framework for LLM-powered multimodal fake news detection in short-form videos.

This repository focuses on the **technical pipeline and experiments**, while the WebApp repository provides the **interactive demonstration interface**.

---

## Notes

* This repository focuses on the technical implementation of the **Refine–Retrieve–Reason** framework.
* The interactive web demonstration is maintained separately.
* Some scripts contain experiment-specific paths and model configurations and may require adjustment before execution in a different environment.
* GPU acceleration is recommended for speech transcription and OCR preprocessing.