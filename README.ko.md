# TRACE: Tracking and Recovery via A Gated Composition of Experts

[English](README.md) | **한국어**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0--beta2-silver)](https://isaac-sim.github.io/IsaacLab/develop/index.html)
[![Physics](https://img.shields.io/badge/physics-Newton-blueviolet)](https://github.com/newton-physics/newton)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**[프로젝트 페이지 및 인터랙티브 데모](https://yester-h.github.io/TRACE/)**

![엎드린 로봇, 누운 로봇, 선 로봇이 모두 일어나 춤을 추는 K1](assets/teaser_run24.gif)

ROBOTIS K1 휴머노이드를 위한 단일 강화학습 정책입니다. 춤을 따라 추다가 넘어지면 누운 자세든 엎드린 자세든,
어느 방향으로 쓰러졌든 스스로 일어나 다시 춤으로 돌아갑니다.
[cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)(Isaac Lab, Newton 백엔드)의 애드온입니다.

## 태스크

| 태스크 ID | 설명 |
| --- | --- |
| `Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand` | 학습 (최종 정책) |
| `Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand-Play` | 평가 및 재생 |

[`trace_recovery/__init__.py`](overlay/source/cyclo_lab/cyclo_lab/manager_based/trace_recovery/__init__.py)에
등록된 나머지 `Cyclo-TRACE-*` 태스크는 이전 버전입니다.

## 0. 사전 설치

[cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)의 Newton 브랜치를 테스트한 커밋으로 받고,
cyclo_lab README를 따라 Docker 컨테이너를 빌드하세요.

```bash
git clone --recursive -b k1/isaacsim-6-newton https://github.com/ROBOTIS-GIT/cyclo_lab.git
cd cyclo_lab && git checkout 5bab742
```

Isaac Sim 6.0.0, Isaac Lab 3.0.0-beta2, Newton 1.2.1, RTX 4060 Laptop GPU(8 GB)에서 테스트했습니다.

## 1. 설치

```bash
git clone https://github.com/YESTER-H/TRACE.git
./TRACE/install.sh /path/to/cyclo_lab
```

`install.sh`는 cyclo_lab의 `train.py` / `play.py`에 작은 패치를 적용한 뒤 [`overlay/`](overlay)의 파일을
cyclo_lab에 복사합니다. 기존 파일은 덮어쓰지 않습니다. 태스크는 `cyclo_lab`을 import할 때 등록됩니다.

## 2. 사용법

cyclo_lab 컨테이너 안의 `/workspace/cyclo_lab`에서 실행합니다.

### 학습

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand --headless \
    --num_envs 4096 --seed 42 --max_iterations 30000
```

`--num_envs 4096`을 꼭 지정하세요(태스크 기본값은 64). RTX 4060 Laptop GPU에서 iteration당 약 2.7초 걸립니다.

### 재생

```bash
CKPT=/path/to/TRACE/checkpoints/run24_gated_prone_trim_yawrand/model_29999.pt
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
    --task Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand-Play \
    --num_envs 3 --checkpoint $CKPT --skip-export
```

### 평가

```bash
TASK=Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand-Play
# 선 자세 / Supine / Prone을 frame 0에서 나란히 시작
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/evaluate_trace_integrated_three_views.py \
    --eval-task $TASK --loop --no-resets --output-dir logs/eval/three_views --checkpoint $CKPT
# 고정 시작 frame에서의 기상 (시작점마다 섭동 32회)
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_fixed_starts.py --headless \
    --eval-task $TASK --supine-frames 0,100,200 --prone-frames 0,100,200 \
    --output-dir logs/eval/fixed_starts --checkpoint $CKPT
# 8방향 × 3단계 속도 밀기
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_push_recovery.py --headless \
    --eval-task $TASK --output-dir logs/eval/push --checkpoint $CKPT
```

각 스크립트는 `--output-dir`에 CSV와 JSON을 저장합니다. 옵션은 `--help`로 확인하세요.

### 인터랙티브 전시

로봇 한 대가 춤추는 동안 공을 던지거나 밀어 볼 수 있습니다.

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/exhibition_trace_push.py \
    --eval-task Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand-Play \
    --output-dir logs/eval/exhibition_$(date +%H%M%S) --checkpoint $CKPT
```

| 키 | 동작 |
| --- | --- |
| `I` `K` `J` `L` / `P` | 뒤 / 앞 / 왼쪽 / 오른쪽 / 무작위 방향 |
| `1` `2` `3` | 세기 |
| `B` | 공 모드(기본) ↔ 밀기 모드 |
| `R` | 춤 시작 자세로 reset |
| `C` | 카메라 따라가기 켜기·끄기 |

### MuJoCo와 브라우저

Isaac Sim 없이 일반 MuJoCo에서도 실행할 수 있습니다.
- [`sim2sim/`](sim2sim): 네이티브 뷰어와 sim2sim 테스트
- [`web/`](web): 브라우저 데모 빌드

## 코드 구조

- **`overlay/source/cyclo_lab/cyclo_lab/manager_based/trace_recovery/`**: TRACE 태스크
  - `commands.py`, `integrated.py`, `gated.py`: 레퍼런스 모션, reset 샘플링, Tracking/Recovery/Transition 로직, gate loss
  - `policy.py`: Tracking/Recovery expert actor
  - `env_cfg.py`, `runner_cfg.py`: 환경과 PPO 설정
- **`overlay/source/cyclo_lab/data/motions/K1_rev1/`**: 춤과 기상 레퍼런스 모션
- **`overlay/scripts/reinforcement_learning/rsl_rl/`**: 평가, 진단, 전시 스크립트
- **`patches/`**: cyclo_lab의 `train.py` / `play.py` 패치
- **`checkpoints/`**: 사전 학습 정책. 최종 정책은 `run24_gated_prone_trim_yawrand`이고, `run23_gated_prone_trim`은 누운 방향 무작위화 없이 학습한 이전 정책입니다.

## 라이선스

Apache License 2.0을 따릅니다. cyclo_lab과 Isaac Lab에서 가져온 코드는 [LICENSE](LICENSE)와
[NOTICE](NOTICE)를 참고하세요. 기상 모션은 BONES-SEED 데이터셋을 리타겟한 것으로 해당 데이터셋의 이용 조건을 따릅니다.
