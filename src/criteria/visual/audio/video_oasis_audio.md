# Video-Oasis Audio-Only Inference Pipeline

## 1. 목적

전체 파이프라인

```text
1. Benchmark JSON + source videos

2. MP4 -> MP3 extraction

3. Whisper STT generation

4. STT noise filtering

5. Text-only LLM inference

6. Prediction / evidence / reasoning JSON
```
=> 각 VQA 문제를 오디오만으로 풀었을 때 **성능 측정** 및 정답의 근거가 되는 STT 타임스탬프를 함께 저장 

## 빠른 실행 요약

```bash
cd Video-Oasis/src/criteria/visual/audio

# 1. extract_audio.py의 JSON_PATH 설정 후
python extract_audio.py

# 2. STT 생성
python run_stt.py --gpus "0,1,2,3" --batch_size 16

# 3. STT 필터링
python filter_stt_noise.py

# 4. run_infer_mp.py의 JSON_PATH와 인증 설정 후
OMP_NUM_THREADS=1 python run_infer_mp.py \
  --gpus "0,1,2,3" \
  --dataset "video-mme" \
  --model_version "qwen25_vl"
```

## 2. 실행 전 전제 조건

### 2.1 작업 디렉터리

- `modules/paths.py`가 현재 clone된 Video-Oasis repo 루트를 자동으로 계산
- `data/benchmarks/audios/*` 입출력은 실행 위치와 관계없이 repo 루트 기준으로 처리
- 데이터셋 JSON처럼 repo 내부 위치가 정해지지 않은 파일만 사용자 환경의 절대경로로 설정

- `data/benchmarks/audios/mp3` (**추가 필요**) : **벤치마크별** 추출된 mp3 저장
- `data/benchmarks/audios/stt` : **벤치마크별** 추출된 stt 저장
- `data/benchmarks/audios/stt_clean`(**추가 필요**) : 벤치마크별 노이즈 필터링 후의 stt 파일 저장

### 2.2 실행 환경

```text
Python 3.10
ffmpeg
CUDA-compatible PyTorch
transformers==4.57.0
accelerate
huggingface_hub
tqdm
flash-attn == 2.6.3
```

```text
torch==2.4.1
torchaudio==2.4.1
torchvision==0.19.1
```

### 2.3 Hugging Face 접근

```text
openai/whisper-large-v3
Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen3-VL-8B-Instruct
nvidia/Eagle2.5-8B
```

### 2.4 경로 설정 관련

**JSON 파일 or mp3/stt 경로는 절대경로로 새로 기입**

```text
extract_audio.py
  JSON_PATH 

run_infer_mp.py
  JSON_PATH
  Hugging Face authentication
```

`run_infer_mp.py`의 Hugging Face 로그인 부분에 유효한 토큰 입력 필요

## 3. 입력 데이터셋 JSON 형식

```json
[
  {
    "question": "In which direction is the turtle walking?",
    "video": "01GfDtSMG4s.mp4",
    "options": [
      "A. Perpendicular to the squirrel from left to right",
      "B. Directly away from the squirrel",
      "C. Perpendicular to the squirrel from right to left",
      "D. Directly towards the squirrel"
    ],
    "answer": "D",
    "answer_text": "Directly towards the squirrel",
    "meta": "Category : Motion and Trajectory Dynamics",
    "db": "ImplicitQA",
    "qid": "1be09cd8-8620-4096-88f6-894bc74b446a",
    "video_path": "/videos/implicitqa/videos/01GfDtSMG4s.mp4"
  }
]
```

## 4. 디렉터리 및 파일명 규칙

의도한 데이터 구조는 다음과 같다.
`data/benchmarks/audios/{mp3/stt}/{데이터셋}`

```text
# 예시
data/benchmarks/audios/
├── mp3/
│   ├── implicitqa/
│   │   └── 01GfDtSMG4s.mp4.mp3
│   └── video-mme/
├── stt/
│   ├── implicitqa/
│   │   └── 01GfDtSMG4s.mp4.mp3.json
│   └── video-mme/
├── stt_clean/
│   ├── implicitqa/
│   │   └── 01GfDtSMG4s.mp4.mp3.json
│   └── video-mme/
└── stt_filtered/
    ├── implicitqa/
    │   └── noisy_video.mp4.mp3.json
    └── video-mme/
```

- stt : 원본 STT
- stt_clean : 노이즈 없는 stt 모음
- stt_filtered : 노이즈가 있어 필터링된 stt 모음 (optional, unnecessary)

stt 또는 stt_clean 설정해서 inference 진행 

## 5. 단계별 실행

### Step 1. MP4에서 MP3 추출
실행:

```bash
cd Video-Oasis/src/criteria/visual/audio

python extract_audio.py
```

+ 데이터셋 json 경로 설정 - EX)
```python
JSON_PATH = "/path/to/vqa_total.json"
```

입출력 예시:

```text
Input
data/benchmarks/videos/video-mme/videos/01GfDtSMG4s.mp4

Output
data/benchmarks/audios/mp3/video-mme/01GfDtSMG4s.mp4.mp3
```

### Step 2. Whisper STT 생성

실행:

```bash
python run_stt.py \
  --gpus "0,1,2,3" \
```

입출력:

1. `data/benchmarks/audios/mp3` 아래의 데이터셋 디렉터리를 자동 탐색
2. 대응하는 `stt/{dataset}` 디렉터리에 JSON을 저장

예시:

```text
Input
data/benchmarks/audios/mp3/video-mme/01GfDtSMG4s.mp4.mp3

Output
data/benchmarks/audios/stt/video-mme/01GfDtSMG4s.mp4.mp3.json
```

STT JSON 형식:

```json
{
  "video_id": "01GfDtSMG4s.mp4.mp3",
  "transcript": "Full transcript text...",
  "timestamps": [
    {
      "timestamp": [0.0, 5.4],
      "text": "Transcript text for this interval."
    }
  ]
}
```


### Step 3. STT 노이즈 필터링

실행:

```bash
python filter_stt_noise.py
```

입출력:

```text
Input    : data/benchmarks/audios/stt/{dataset}/*.json
Clean    : data/benchmarks/audios/stt_clean/{dataset}/*.json
Filtered : data/benchmarks/audios/stt_filtered/{dataset}/*.json
```

### Step 4. Audio-only LLM 추론

단일 데이터셋 및 모델 실행:

```bash
OMP_NUM_THREADS=1 python run_infer_mp.py \
  --gpus "0,1,2,3" \
  --dataset "video-mme" \
  --model_version "qwen25_vl"
```

추론 프롬프트는 다음 정보만 모델에 제공한다.

출력 디렉터리:

```text
output_prior/{model_version}/{dataset}_result.json
```

출력 객체 예시:

```json
{
  "qid": "1be09cd8-8620-4096-88f6-894bc74b446a",
  "video": "01GfDtSMG4s.mp4",
  "db": "ImplicitQA",
  "question": "In which direction is the turtle walking?",
  "options": [
    "A. ...",
    "B. ...",
    "C. ...",
    "D. ..."
  ],
  "ground_truth": "D",
  "predicted_option": "D",
  "is_correct": true,
  "Evidence": "[12.0 - 15.3] ...",
  "Reasoning": "..."
}
```

응답 파서는 실제 보기 개수에 따라 `A`부터 마지막 보기 문자까지 허용한다. 모델 출력이 형식에 맞지 않거나 유효 범위 밖이면 `Unknown`으로 저장한다.

* 전체 모델과 데이터셋 조합 실행 (background)

```bash
chmod +x run_all_inference.sh
./run_all_inference.sh
```

로그 저장 위치 

```text
logs/{model}_{dataset}_{timestamp}.log
```

## 6. 현재 코드 기준 실행 전 확인사항

- 현재 오디오 추출 단계는 JSON의 `db`를 소문자로 변환하여 디렉터리를 만들고 있으나, 이 경우 MMR-V 벤치마크는 `mmr-v`로 변환된다. 하지만 github repo 기준 MMR-V 벤치를 `mmrvbench`로 정의하고 있어, 벤치마크 이름을 **mmrvbench -> mmr-v**로 정의할 것을 제안 
