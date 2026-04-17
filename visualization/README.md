---

# Embedding Analysis & Visualization

## Overview

본 노트북은 텍스트, 생성, 오디오, 그리고 fusion 임베딩 간의 관계를 분석하고 시각화하기 위한 도구이다.
PCA, UMAP, 코사인 유사도 기반 분석을 통해 각 임베딩의 구조적 특성과 정렬 관계를 확인한다.

---

## Features

### 1. Embedding Visualization (PCA / UMAP)

* 텍스트 임베딩 PCA 시각화
* 텍스트 임베딩 UMAP 시각화
* 플레이리스트별 분포 및 군집 구조 확인

---

### 2. Embedding Comparison

* 텍스트 vs 생성 임베딩 비교 (단일 플레이리스트)
* 오디오 vs 생성 임베딩 비교 (다중 플레이리스트)
* 오디오 vs 생성 vs fusion 임베딩 비교

---

### 3. Cosine Similarity Analysis

* Text ↔ Generated
* Generated ↔ Audio
* Text ↔ Audio
* 트랙 단위 코사인 유사도 계산

---

### 4. Bridge Pattern Visualization

* x축: Text–Generated
* y축: Generated–Audio
* 색상: Text–Audio
* 생성 임베딩이 텍스트와 오디오 사이를 연결하는지 분석

---

### 5. Delta Distribution Analysis

```text
Δ = cos(Text, Generated) - cos(Text, Audio)
```

* Δ 값 분포를 histogram으로 시각화
* 생성 임베딩의 정렬 개선 여부 확인

---

### 6. Sampling & Filtering

* 플레이리스트 단위 데이터 선택
* 샘플 수 제한을 통한 효율적인 분석 수행

---

### 7. Figure Export

* 모든 시각화 결과를 `.png` 파일로 저장
* 논문 및 발표 자료에 활용 가능

---

## Input Data

각 샘플은 다음 정보를 포함해야 한다:

* `text_emb_path`
* `gen_path`
* `audio_path`
* `gate_path` (optional)
* `plylst_id`

모든 임베딩은 `.npy` 형식의 동일 차원 벡터여야 한다.

---

## Pipeline

```text
CSV 로드 → 임베딩 로드 → 전처리/샘플링 →
PCA/UMAP/유사도 계산 → 시각화 → 결과 저장
```

---

## Purpose

* 멀티모달 임베딩 간 정렬 관계 분석
* 생성 임베딩의 bridge 역할 검증
* fusion representation 구조 분석
* qualitative 결과 시각화

---
