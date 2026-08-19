# Object-Centric Perception 모듈 — 블록 단위 설계 계획

## Context

LIBERO-plus(10,030 태스크, 7개 perturbation 차원) 벤치마크에서 X-VLA baseline은 현재 별도로 평가가 진행 중이며(참고용 수치), PROJECT_OVERVIEW.md의 목표는 **perturbation 데이터 학습 없이 아키텍처만으로** 95% 이상 성공률을 달성하는 것이다. 제안된 해법은 SAM3(frozen) 기반 object 인식 → depth 3D localization → EEF 상대좌표 → 언어-물체 cross-attention/FiLM → set-invariant encoder를 거쳐 X-VLA policy에 주입하는 파이프라인이다.

본 계획은 이 파이프라인을 **기능 단위 블록**으로 쪼개고, X-VLA 실제 소스코드(`models/transformer.py`, `modeling_xvla.py`, `action_hub.py`, `peft_train.py`, `datasets/domain_handler/`)를 직접 읽어 확인한 통합 지점을 기준으로 **정확한 구현 순서**를 정한다. 아직 코드는 작성하지 않는다 — 이 문서는 실행 전 합의를 위한 설계도다.

---

## 0. 인터페이스 계약 (구현 전 반드시 확정할 결정 사항)

이 결정들이 이후 모든 블록의 텐서 shape/의존성을 고정하므로 코딩 착수 전 확정한다.

| 결정 | 선택 | 근거 |
|---|---|---|
| Cross-Attention용 언어 토큰 소스 | Florence2 **토크나이저 임베딩 lookup** (`self.vlm.get_input_embeddings()(input_ids)`), frozen | `forward_vlm()`의 `vlm_features`(post-encoder, 이미지와 이미 융합됨)를 쓰면 Phase 2(Fusion)가 Phase 3(정책 통합)에 종속되어 버림. Raw embedding을 쓰면 Phase 2가 SAM3/sim 없이 synthetic tensor만으로 완전 독립 단위테스트 가능 |
| Object 토큰 최종 개수 | 고정 K (예: K=8) | `transformer.py`의 `pos_emb`가 고정 인덱스 파라미터라 가변 길이 세그먼트를 넣을 수 없음 → Set-invariant encoder가 반드시 고정 K개 출력을 내야 함 |
| Policy 주입 방식 | `vlm_proj`/`aux_visual_proj`와 병렬인 **3번째 concat 스트림** (`object_proj`) | 대안(= aux_visual_inputs에 합산)은 K가 aux patch 개수와 강제로 같아져야 해서 고정-K 설계 목적을 무너뜨림 |
| Object 토큰용 위치 임베딩 | `pos_emb`의 미사용 tail을 재사용하지 **않고**, `soft_prompt_hub`와 동일한 패턴의 전용 `nn.Parameter(K, hidden_size)` 신설 | tail 재사용은 의미 없는 위치 bias를 주고, 시퀀스 길이가 늘어나면 인덱스 충돌 위험 |
| 학습 시 object 토큰 계산 방식 | **오프라인 캐시** (episode·frame 키로 1회 계산 후 디스크 저장) | SAM3 zero-shot·frozen이라 매 epoch 재계산할 이유 없음. 단, 캐시 경계는 **Perception(블록 1~5) 출력까지만** — Fusion(블록 6~8, 10)은 학습 가능한 모듈이므로 매 forward마다 라이브로 실행 |
| Depth 소스 | 학습: HDF5 `states` replay로 재생성 / 추론: `OffScreenRenderEnv(camera_depths=True)` 실시간 | 원본 LIBERO HDF5, X-VLA 재가공 HDF5, `lerobot/libero` 어디에도 depth 없음 (직접 확인) |
| 언어-무관 물체의 처리 | **제거하지 않고 가중치만 다르게** (블록9 Threshold+제거 폐기, 2026-08-19 변경) | 정책 입력이 이미지가 아니라 상대좌표뿐이라, 언어와 무관한 물체를 완전히 제거해버리면 그게 경로를 막는 장애물이어도 정책이 인식할 방법이 없음 (예: "white cup을 yellow bowl에 놓아라"인데 blue cup이 경로 중간에 있는 경우). 블록7(관련도 계산)+블록8(FiLM 가중치)은 유지하되, 블록9는 파이프라인에서 제거 — 물체 인지(장애물 회피용)와 물체 관련도(과업 수행용)를 분리 |
| 물체 크기 표현 (2026-08-19 추가) | **3축 반너비 스칼라 3개** (PCA, 방향 벡터 없이 크기만) — 점 하나(구체 근사)로는 회피 시 "얼마나 피해야 하나"를 모름. 전체 boundary/contour는 (a) 한 카메라 시점의 부분 실루엣일 뿐이고 (b) 물체마다 개수가 달라 고정 크기 토큰에 안 맞고 (c) 계산·정보 복잡도만 늘어남 → 기각 | 파지 방향까지는 이 크기 정보 + 블록6의 의미 라벨 임베딩("bowl"이 이미 형태 prior를 내포) + X-VLA의 사전학습된 foundation 지식에 맡김. 별도 grasp-pose 추정 모듈은 만들지 않음 — 원래 계획서에도 없던 범위이며, 평가(§5.3 실패 분류에 "파지/접촉 실패" 범주 추가 예정)로 실제 필요성이 확인되면 그때 추가 |

---

## Phase 1 — Perception 블록 (정적 이미지/replay로 독립 테스트 가능)

문서의 파이프라인 순서(SAM3 → clustering → depth → EEF상대좌표)를 그대로 따르되, 명사구 추출이 SAM3 입력이므로 가장 먼저 와야 한다.

| # | 블록 | 파일 | 입력 → 출력 | 의존성 | 검증 |
|---|---|---|---|---|---|
| 1 | NP(명사구) 추출 | `perception/np_extractor.py` | 명령어 문자열 → 명사구 리스트 | 없음 (기존 파서/LLM 재사용, 신규 학습 없음) | 샘플 명령어 10개 육안 확인 |
| 2 | SAM3 wrapper | `perception/sam3_wrapper.py` | RGB + 명사구 리스트 → per-object mask/bbox/conf | #1, **SAM3 설치 필요(미설치 확인됨)** | LIBERO 렌더링 이미지에 대한 mask 품질 육안 확인 (문서 §6 주의사항) |
| 3 | 물체 클러스터링 | `perception/clustering.py` | mask들 → 대표 pixel 좌표 | #2 | distractor 밀도 높은 프레임에서 grounding 정확도 사전 점검 |
| 4 | Depth→3D localization + 크기 추정 | `perception/depth_localize.py` | 대표 pixel + depth + camera intrinsic/extrinsic → 3D 좌표(world frame); mask 전체 pixel → point cloud → PCA 3축 반너비(`estimate_object_extent`) | #3, depth 소스 (아래 4b) | 왕복 투영 오차 확인 + bowl(지름~10cm,높이~5cm)/plate(지름~13cm,두께~1.8cm) 등 실측 대비 형태 타당성 확인 완료 |
| 4b | **[선행 필요]** Depth 재생성 (학습용) | `data/regenerate_depth.py` | 원본 LIBERO demo HDF5의 `states` → replay → depth map | 없음 (블록 12와 동일 스크립트, 순서상 여기 먼저 실행) | 아래 "Depth 재생성 검증" 참고 |
| 5 | EEF 상대좌표 변환 | `perception/relative_pose.py` | 3D 좌표 + 현재 EEF pose → egocentric 상대좌표 (매 타임스텝 갱신) | #4 | 정적 물체가 EEF 이동에 따라 상대좌표만 바뀌는지 시뮬레이션 확인 |

**EEF pose 소스**: 새로 만들 필요 없음. `evaluation/libero/libero_client.py`가 이미 매 스텝 `env.env.robots[0].controller.ee_pos` / `ee_ori_mat`(→6D)을 계산하고 있음 — 학습(replay) 시와 추론(live env) 시 동일 소스를 재사용.

**카메라 파라미터**: `robosuite.utils.camera_utils.get_camera_intrinsic_matrix`, `get_camera_extrinsic_matrix`, `get_real_depth_map` 존재 확인 — 직접 구현 불필요.

⚠️ **eye_in_hand 카메라 주의**: extrinsic이 매 프레임 그리퍼 pose에 따라 바뀜 — 정적 캐싱 금지, 프레임마다 재계산.

---

## Phase 2 — Fusion 블록 (순수 텐서 연산, synthetic tensor로 SAM3/sim 없이 단위테스트 가능)

Phase 1 출력 스키마 확정 후 시작. 블록 간 순서는 엄격히 순차적 (F1→F2→F3→F4→F5).

| # | 블록 | 파일 | 핵심 로직 |
|---|---|---|---|
| 6 | Projection/Adapter | `fusion/projection.py` | object 토큰(3D pos + mask 특징) → 공통 embedding dim |
| 7 | Cross-Attention | `fusion/cross_attention.py` | 언어 토큰(§0 결정: frozen embedding) × #6 출력 → 관련도 score |
| 8 | FiLM | `fusion/film.py` | #7 score → scale/shift 생성 → object embedding에 적용 (관련도 높은 물체를 증폭하되, 낮은 물체도 제거하지 않고 남김) |
| ~~9~~ | ~~Threshold 필터링~~ | ~~`fusion/threshold.py`~~ | **폐기 (2026-08-19)** — 물체 제거는 안 함, §0 "언어-무관 물체의 처리" 참고 |
| 10 | Set-invariant encoder | `fusion/set_encoder.py` | **가변 개수 전체(필터링 없음)** → 고정 K개 pooled 토큰 (attention readout) |

---

## Phase 3 — Policy 통합 (X-VLA 백본 연결)

**핵심 파일**: `policy/xvla_adapter.py` (신규) — 벤더 코드(`models/*.py`)를 직접 수정하지 않고 **subclass/wrap**하는 방식으로 구현 (되돌리기/비교 용이).

구체적 통합 지점 (실제 코드 확인 완료):
- `models/transformer.py:319-324` — `vlm_proj`/`aux_visual_proj`와 동일한 패턴으로 `object_proj = nn.Linear(fusion_dim, hidden_size)` 추가
- `models/transformer.py:376-383` — `x = cat([action_tokens, vlm_proj(...), aux_visual_proj(...)], dim=1)`에 `object_proj(object_tokens)`를 3번째 스트림으로 추가
- `models/transformer.py:326-327` — `pos_emb`는 건드리지 않고, `soft_prompt_hub`(L335-337)와 동일 패턴의 전용 `object_pos_emb = nn.Parameter(1, K, hidden_size)` 신설
- `models/modeling_xvla.py:104-145` (`forward_vlm`) — 변경 없음 (Florence2 언어/이미지 인코더는 그대로 frozen 재사용)

**Fine-tune 범위** (문서 §3 표 반영):

| 구성요소 | 처리 |
|---|---|
| Florence2 (`self.vlm`) | Frozen (전체) |
| `self.transformer.blocks`(24층) | 대부분 frozen, 1일 프로토타입은 앞/뒤 2~4개 층만 학습 |
| 신규 모듈(블록 6,7,8,10,11: projection/cross-attn/FiLM/set-encoder/object_proj) | Full train |
| `action_encoder`/`action_decoder` | Fine-tune (기존과 동일) |
| Backbone 전체 LoRA | 본 실험(1단계) 단계에서만 검토 |

**학습 스크립트 통합**: `peft_train.py:118-131`의 `build_optimizer()`가 이미 vlm/transformer_core/soft_prompts/action_heads로 param group을 나누는 패턴을 갖고 있음 → 새 param group `"perception_fusion"` (블록 6~11 파라미터, 항상 full lr)을 추가하는 방식으로 확장. 완전히 새 학습 스크립트를 만들 필요 없음.

---

## Phase 4 — 데이터 준비

| # | 블록 | 파일 | 비고 |
|---|---|---|---|
| 4b/12 | Depth 재생성 | `data/regenerate_depth.py` | `evaluation/libero/rel2abs.py`의 replay 패턴(HDF5 `states` → `env.set_init_state` → `env.step` loop)을 그대로 재사용하되 `env_args`에 `camera_depths=True` 추가 |
| 13 | 학습 메타 준비 | `data/prepare_libero.py` | `datasets/domain_handler/simulations.py`의 기존 `LiberoHandler`(HDF5 기반, `abs_action_6d` 스키마)를 참고해 depth 포함 버전 handler로 확장 — `lerobot/libero`(HF 포맷)이 아니라 X-VLA가 실제로 쓰는 재가공 HDF5 포맷 기준으로 맞춤 |

⚠️ **정합성 위험 (반드시 검증)**: `LiberoHandler.get_image_datasets()`(simulations.py:105-109)가 "image desync quirk" 때문에 첫 프레임을 drop함. depth 재생성 스크립트가 이 오프셋을 정확히 맞추지 않으면 RGB[i]와 depth[i]가 조용히 한 프레임씩 어긋남(에러 없이 학습 품질만 저하). → **재생성된 depth 프레임을 이미 저장된 RGB 프레임과 index 맞춰 pixel-diff로 스팟체크하는 assertion을 재생성 스크립트에 필수 포함.**

---

## Phase 5 — 학습

| # | 블록 | 내용 |
|---|---|---|
| 14 | 학습 루프 확장 | `peft_train.py` 기반, `xvla_adapter.py` 모델 + 신규 param group 삽입. 캐시된 Phase 1 출력(§0)을 배치에 결합하는 collate 로직 추가 |
| — | 스모크 테스트 | 수십~백 개 샘플로 1배치 forward/backward, loss가 NaN 없이 감소하는지 확인 (본 학습 착수 전 필수 게이트) |

---

## Phase 6 — 평가

| # | 블록 | 내용 |
|---|---|---|
| 15 | `eval/proposed_eval.py` | 기존에 이미 검증된 `deploy.py` + `evaluation/libero/libero_client.py` client-server 패턴 재사용 (현재 세션에서 LIBERO-plus 대상 실제 동작 확인됨). 모델만 `xvla_adapter.py` 체크포인트로 교체 |
| 16 | `eval/failure_analysis.py` | Object Layout 결과를 방해물체 추가 vs 타겟 재배치로 분리 집계, 실패를 grounding/OOD/참조모호성 3종으로 분류 (문서 §5.3) |
| — | baseline_eval | 현재 진행 중인 LIBERO-plus X-VLA 평가 결과를 그대로 재사용 (별도 재실행 불필요) |

---

## 전체 실행 순서 요약

```
[Phase 0: 인터페이스 계약 확정 — 문서화만, 코드 없음]
        │
[Phase 1: 1→2→3→4b→4→5]  (Perception, 순차적 — 각 블록이 이전 블록 출력에 의존)
        │
[Phase 2: 6→7→8→10]       (Fusion, Phase 1과 병렬 착수 가능 — synthetic tensor로 독립 개발; 블록9 폐기)
        │
[Phase 3: 11 (xvla_adapter.py)]   ← Phase 1 + Phase 2 산출물이 합쳐지는 지점
        │
[Phase 4: 12→13]          (데이터 준비 — Phase 1의 4b와 동일 스크립트, 학습 착수 전 완료되어야 함)
        │
[Phase 5: 14 → 스모크테스트]
        │
[Phase 6: 15→16]
```

**병렬화 가능 지점**: Phase 1(Perception)과 Phase 2(Fusion)는 서로 독립적으로 동시 개발 가능 (§0에서 인터페이스를 미리 고정했기 때문). SAM3 설치가 막히더라도 Phase 2는 synthetic 텐서로 계속 진행 가능.

---

## 디렉토리 구조 (파일별 핵심 코드 요약)

```
project/
├── perception/
│   ├── np_extractor.py        # [블록1] 명령어 → 명사구 리스트
│   ├── sam3_wrapper.py        # [블록2] SAM3 추론, text prompt → mask/bbox
│   ├── clustering.py          # [블록3] mask → 대표 pixel 좌표
│   ├── depth_localize.py      # [블록4] robosuite camera_utils 활용, pixel+depth → 3D
│   └── relative_pose.py       # [블록5] 3D → EEF 기준 상대좌표 (매 스텝 갱신)
├── fusion/
│   ├── projection.py          # [블록6] object 토큰 → 공통 embedding dim
│   ├── cross_attention.py     # [블록7] 언어(frozen embedding) × object 토큰
│   ├── film.py                # [블록8] score → scale/shift (제거 없이 가중치만)
│   └── set_encoder.py         # [블록10] 가변개수(필터링 없는 전체) → 고정 K 토큰
├── policy/
│   └── xvla_adapter.py        # [블록11] XVLA/SoftPromptedTransformer subclass
│                               #   - object_proj 추가 (transformer.py:319-324 패턴)
│                               #   - concat 지점 확장 (transformer.py:376-383)
│                               #   - object_pos_emb 신규 파라미터 (soft_prompt_hub 패턴)
├── data/
│   ├── regenerate_depth.py    # [블록4b/12] rel2abs.py 패턴 재사용, depth 재생성
│   └── prepare_libero.py      # [블록13] LiberoHandler 확장, 4-suite 메타 구성
├── eval/
│   ├── proposed_eval.py       # [블록15] libero_client.py 패턴 재사용
│   └── failure_analysis.py    # [블록16] Object Layout 하위유형 분류
└── PROJECT_OVERVIEW.md         # (아직 디스크에 없음 — 아래 참고)
```

**기존 X-VLA 코드 중 재사용/참조할 핵심 파일**:
- `X-VLA/models/transformer.py` — `SoftPromptedTransformer` (통합 지점, L341-403)
- `X-VLA/models/modeling_xvla.py` — `forward_vlm()`(L104-145), `generate_actions()`(L181-216)
- `X-VLA/models/action_hub.py` — `EE6DActionSpace`(gripper_idx=(9,19), dim_action=20)
- `X-VLA/peft_train.py` — `build_optimizer()`(L118-131), 학습 루프 전체
- `X-VLA/datasets/domain_handler/simulations.py` — `LiberoHandler`(L97-121)
- `X-VLA/evaluation/libero/rel2abs.py` — HDF5 replay 패턴 (depth 재생성 템플릿)
- `X-VLA/evaluation/libero/libero_client.py` — EEF pose 추출 방식(`controller.ee_pos`), eval client-server 패턴

**참고**: `PROJECT_OVERVIEW.md`는 현재 대화에서 텍스트로만 제공되었고 디스크(`X-VLA/` 루트)에는 실존하지 않음 — 문서 자체 지침("이 문서를 프로젝트 루트에 두고 작업 시작")대로 실제 파일로 저장할지 별도 확인 필요.

---

## 검증 방법 (블록별 "완료" 기준)

- **블록 1~5 (Perception)**: LIBERO 렌더링 이미지 5~10장에 대해 mask/3D좌표/상대좌표를 시각화해 육안 검증 (문서 §6 주의사항과 동일)
- **블록 6~8, 10 (Fusion)**: synthetic 텐서(랜덤 object 개수 0~15개)로 forward pass가 shape 에러 없이 통과하고, 모든 입력·파라미터에 gradient가 흐르는지(`loss.backward()` 후 각 파라미터 `.grad`가 None이 아닌지) 확인. 블록9는 폐기되어 검증 대상에서 제외 — §0 "언어-무관 물체의 처리" 참고
- **블록 11 (xvla_adapter)**: 1배치 forward, `pred_action` shape이 원본 X-VLA와 동일한지, 원본 대비 추가 파라미터 수 확인
- **블록 12 (Depth 재생성)**: 재생성 RGB[i] vs 저장된 원본 RGB[i] pixel-diff 스팟체크 (프레임 정합성 확인 필수)
- **블록 14 (학습 스모크테스트)**: 수십~백 샘플, 수십 step, loss NaN 없이 감소
- **블록 15~16 (평가)**: 현재 진행 중인 baseline 대비 선택 perturbation 부분집합에서 성공률 비교 + attention map이 언어 관련 물체에 가중치를 싣는지 정성 확인 (문서 §6 성공 기준)
