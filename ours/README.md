## 개요

본 프로젝트는 생성 기반 멀티모달 플레이리스트 추천 모델의 학습 및 평가 코드를 재현 가능하도록 정리한 결과물이다.  
입력 데이터로는 각 트랙에 대해 미리 추출된 **원본 오디오 임베딩(audio embedding)** 과 **텍스트 조건 기반 생성 임베딩(generated embedding)** 을 사용하며, 이 두 표현을 게이트 기반으로 융합한 뒤 Transformer encoder를 통해 플레이리스트 문맥을 학습한다.



## 프로젝트 파일 구성

단일 스크립트로 작성되어 있던 코드를 기능별로 분리한 구조는 다음과 같다.

```text
project/
├── main.py
├── model_parts.py
├── data_utils.py
├── sampling.py
├── train_eval.py
├── config.py
├── dataset.py
└── README.md
```

### 파일별 역할

#### `main.py`
- 전체 학습 및 평가 실행 진입점
- 설정 로드
- 데이터셋 / DataLoader 생성
- 모델 생성
- optimizer 설정
- 카탈로그 로드
- epoch 단위 학습 및 sampled metric 평가
- 마지막 epoch에서 checkpoint 저장

#### `model_parts.py`
- 모델 본체 정의
- `PerStepGate`
- `build_causal_mask`
- `RecommenderModel`

#### `data_utils.py`
- 배치 구성 관련 함수
- `collate_fn`
- 디버그 출력 함수
- 게이트 엔트로피 loss 함수
- coverage 확인 함수

#### `sampling.py`
- sampled evaluation용 후보 샘플링 로직
- negative candidate 생성
- recall / nDCG / MRR 계산에 필요한 보조 함수

#### `train_eval.py`
- 학습 및 평가 함수 정의
- `run_epoch_onehot`
- `compute_full_recall`
- `compute_sampled_metrics`

#### `config.py`
- 학습 설정(dataclass) 정의
- 데이터 경로, 배치 크기, 학습률, epoch 수, 저장 경로 등 관리

#### `dataset.py`
- 데이터 로딩 정의
- `PlaylistDataset`
- `build_catalog_from_json`

---

## 데이터 활용 방식

### 입력 데이터 개요

본 모델은 raw audio를 직접 입력받지 않고, **사전 추출된 1차원 임베딩 벡터(.npy)** 를 입력으로 사용한다.

각 학습 샘플은 하나의 플레이리스트 시퀀스로 구성되며,
- 마지막 트랙은 정답(target track)
- 그 이전 트랙들은 문맥(context track)
으로 사용된다.

문맥 트랙마다 다음 두 종류의 임베딩이 필요하다.

- `audio_path`: 원본 오디오 임베딩 경로
- `gen_path`: 텍스트 조건 기반 생성 임베딩 경로

정답 트랙에 대해서는 다음 정보가 사용된다.

- `audio_path`: 정답 오디오 임베딩 경로
- `track_id`: 정답 아이템 ID

즉, 데이터는 **플레이리스트 단위 시퀀스 정보 + 트랙별 임베딩 파일 경로** 로 구성된다.

---

### 학습용 JSON 구조

학습/평가 JSON은 대략 아래와 같은 레코드들의 리스트 형태로 구성된다.

```json
[
  {
    "playlist_id": 1001,
    "track_id": 55501,
    "audio_path": "/path/to/audio_emb/55501.npy",
    "gen_path": "/path/to/gen_emb/55501.npy"
  },
  {
    "playlist_id": 1001,
    "track_id": 55502,
    "audio_path": "/path/to/audio_emb/55502.npy",
    "gen_path": "/path/to/gen_emb/55502.npy"
  }
]
```

같은 `playlist_id` 를 가진 항목들이 하나의 시퀀스를 이룬다.  
`PlaylistDataset` 은 이를 playlist별로 묶은 뒤, 길이가 2 이상인 경우만 학습 샘플로 사용한다.

---

### 카탈로그 JSON 구조

카탈로그 파일은 전체 후보 아이템 임베딩을 구성하기 위해 사용된다.

```json
[
  {
    "track_id": 55501,
    "audio_path": "/path/to/audio_emb/55501.npy"
  },
  {
    "track_id": 55502,
    "audio_path": "/path/to/audio_emb/55502.npy"
  }
]
```

이 카탈로그는 전체 아이템 집합에 대한 분류형 softmax와 sampled metric 계산에 사용된다.


## `config.py` 설명

아래설정 클래스는 실험 재현을 위한 핵심 하이퍼파라미터와 경로를 정의한다.


### 주요 설정 의미

- `train_json`, `val_json`, `test_json`: 학습/검증/테스트 시퀀스 JSON 경로
- `catalog_json`: 전체 후보 아이템 카탈로그 경로
- `d`: 임베딩 차원 수
- `batch_size`: 학습 배치 크기
- `num_workers`: DataLoader worker 수
- `lr`: AdamW 학습률
- `epochs`: 총 학습 epoch 수
- `weight_decay`: 가중치 감쇠 값
- `grad_clip`: gradient clipping 값
- `alpha_retr`: 보조 InfoNCE loss 가중치
- `lambda_gate_entropy`: 게이트 붕괴 방지 정규화 강도
- `seed`: 재현성 확보를 위한 랜덤 시드
- `device`: 학습 장치 (`cuda` 또는 `cpu`)
- `save_dir`: 체크포인트 저장 경로

---

## `dataset.py` 설명

`PlaylistDataset` 은 JSON 파일을 읽어 playlist 단위로 묶고, 각 playlist를 다음과 같이 하나의 학습 샘플로 변환한다.

### 입력 처리 방식

- 같은 `playlist_id` 를 가진 항목들을 하나의 시퀀스로 그룹화
- 시퀀스 마지막 항목을 target track으로 사용
- 그 이전 항목들을 context track으로 사용

### 반환되는 항목

```python
{
    "e_a": 원본 오디오 임베딩 시퀀스 (L, d),
    "e_t": 생성 임베딩 시퀀스 (L, d),
    "e_pos": 정답 트랙 오디오 임베딩 (d,),
    "pid": 플레이리스트 ID,
    "pos_id": 정답 트랙 ID,
    "ctx_ids": 문맥 트랙 ID 목록,
    "len": 문맥 길이
}
```

### 카탈로그 로딩

`build_catalog_from_json()` 함수는 전체 후보 아이템의 audio embedding을 불러와 `(N, d)` 형태 텐서로 구성한다.

---

## 모델 구조 설명

### 입력 표현

각 플레이리스트 문맥 시점마다 두 개의 임베딩이 입력된다.

- `e_a`: 원본 오디오 임베딩
- `e_t`: 생성 임베딩

### 게이트 융합

`PerStepGate` 는 각 시점에서 두 임베딩을 입력받아 0~1 사이의 게이트 값을 출력한다.

```text
h = (1 - g) * e_t + g * e_a
```

즉, 시점별로 생성 임베딩과 오디오 임베딩의 비중을 동적으로 조절한다.

### Transformer encoder

융합된 시퀀스는
- LayerNorm
- Linear projection
- learnable positional embedding
- causal Transformer encoder
를 거쳐 시퀀스 문맥 표현으로 변환된다.

### 최종 표현

각 샘플의 마지막 유효 시점 벡터를 가져와 head MLP에 통과시킨 후 residual connection을 적용하고, 최종적으로 L2 normalize된 추천 표현 `z` 를 얻는다.

---

## 학습 방식

본 코드는 **full catalog one-hot cross entropy** 기반 학습을 사용한다.

### 기본 loss

정답 트랙의 카탈로그 인덱스를 label로 두고,

```text
logits = (q · V^T) / tau
```

형태의 전체 아이템 분류 문제로 학습한다.

- `q`: 모델이 예측한 playlist context representation
- `V`: 전체 카탈로그 임베딩
- `tau`: learnable temperature

### 보조 loss

옵션으로 다음 항목이 추가된다.

#### Retrieval InfoNCE
정답 임베딩과의 정렬을 강화하기 위한 보조 cross entropy

#### Gate entropy regularization
게이트 값이 한쪽 모달리티로 과도하게 붕괴되는 현상을 완화하기 위한 정규화 항

---

## 평가 방식

### Sampled evaluation

전체 카탈로그에서 정답 1개와 negative candidate 여러 개를 샘플링하여 다음 지표를 계산한다.

- Recall@10
- Recall@20
- Recall@50
- NDCG@10
- NDCG@20
- NDCG@50
- MRR

### Context 제거 옵션

평가 시, 필요에 따라 context에 이미 포함된 track을 negative 후보에서 제외할 수 있도록 구성되어 있다.
