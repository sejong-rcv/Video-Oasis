# Video-Oasis Audio-Only Inference Pipeline

## 1. 목적

이 디렉터리는 비디오의 시각 정보 없이 오디오에서 추출한 STT만 사용하여 VQA 객관식 문제를 푸는 논문 재현 코드를 포함한다.

전체 파이프라인은 다음과 같다.

```text
Benchmark JSON + source videos
        |
        v
MP4 -> MP3 extraction
        |
        v
Whisper STT generation
        |
        v
STT noise filtering
        |
        v
Text-only LLM inference
        |
        v
Prediction / evidence / reasoning JSON
```

주요 목적은 각 VQA 문제를 오디오 정보만으로 풀었을 때의 성능을 측정하고, 정답 근거가 되는 STT 타임스탬프를 함께 저장하는 것이다.

## 2. 실행 전 전제 조건

### 2.1 작업 디렉터리

현재 스크립트의 데이터 경로는 상대 경로로 작성되어 있다. 아래 명령을 실행하는 위치를 기준으로 `data/benchmarks/audios`가 해석된다.

```bash
cd Video-Oasis/src/preprocess/audio
```

따라서 현재 코드 그대로 실행할 경우 데이터 디렉터리는 다음 위치에 있어야 한다.

```text
Video-Oasis/src/preprocess/audio/data/benchmarks/audios/
```

저장소 루트의 `Video-Oasis/data/benchmarks`를 사용하려면 각 스크립트의 경로를 저장소 루트 기준으로 수정해야 한다.

### 2.2 실행 환경

필요한 주요 프로그램과 Python 패키지는 다음과 같다.

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

기존 실험 환경에 기록된 PyTorch 계열 버전은 다음과 같다.

```text
torch==2.4.1
torchaudio==2.4.1
torchvision==0.19.1
```

실제 설치 버전은 CUDA 버전 및 사용하는 Qwen/Eagle 모델과 호환되어야 한다. LLM 추론 코드는 `flash_attention_2`와 `bfloat16`을 사용하므로 이를 지원하는 GPU 및 라이브러리 구성이 필요하다.

### 2.3 Hugging Face 접근

다음 모델을 다운로드할 수 있어야 한다.

```text
openai/whisper-large-v3
Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen3-VL-8B-Instruct
nvidia/Eagle2.5-8B
```

`run_infer_mp.py`의 Hugging Face 로그인 부분에는 유효한 토큰 또는 별도의 인증 방식이 필요하다. 토큰을 Git에 직접 커밋하지 않는다.

## 3. 입력 데이터셋 JSON 형식

JSON 파일의 최상위 값은 QA 객체들의 리스트여야 한다.

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

하나의 비디오에 여러 질문이 있을 수 있다. 오디오 추출 단계에서는 `(db.lower(), video_path)` 조합으로 중복을 제거하여 비디오당 MP3 하나만 생성한다.

## 4. 디렉터리 및 파일명 규칙

의도한 데이터 구조는 다음과 같다.

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


파일명 변환 규칙은 다음과 같다.

```text
Source video : 01GfDtSMG4s.mp4
MP3          : 01GfDtSMG4s.mp4.mp3
STT          : 01GfDtSMG4s.mp4.mp3.json
```

원본 비디오 확장자를 제거하지 않고 전체 파일명 뒤에 `.mp3`를 추가한다. STT 파일은 MP3 파일명 전체 뒤에 `.json`을 추가한다.

JSON의 `db` 값은 오디오 추출 시 소문자로 변환된다.

```text
ImplicitQA -> implicitqa
Video-MME  -> video-mme
```

이후 모든 단계에서 동일한 소문자 데이터셋 디렉터리 이름을 사용해야 한다.

## 5. 단계별 실행

### Step 1. MP4에서 MP3 추출

관련 파일:

```text
extract_audio.py
modules/audio_utils.py
modules/data_loader.py
```

먼저 `extract_audio.py`의 `JSON_PATH`를 전체 QA JSON 경로로 설정한다.

```python
JSON_PATH = "/path/to/vqa_total.json"
```

실행:

```bash
python extract_audio.py
```

동작:

1. JSON 리스트를 로드한다.
2. 각 항목에서 `db`와 `video_path`를 읽는다.
3. `(db.lower(), video_path)` 단위로 중복 비디오를 제거한다.
4. 16개 CPU worker를 사용해 ffmpeg를 실행한다.
5. 데이터셋별 MP3 디렉터리에 결과를 저장한다.

ffmpeg 변환 설정:

```text
Audio channel : mono
Sampling rate : 16 kHz
Codec         : libmp3lame
Quality       : -q:a 4
```

예시:

```text
Input
/videos/implicitqa/videos/01GfDtSMG4s.mp4

Output
data/benchmarks/audios/mp3/implicitqa/01GfDtSMG4s.mp4.mp3
```

이미 출력 파일이 존재하면 해당 비디오는 건너뛴다.

### Step 2. Whisper STT 생성

관련 파일:

```text
run_stt.py
modules/stt_utils.py
```

실행:

```bash
python run_stt.py \
  --gpus "0,1,2,3" \
  --batch_size 16 \
  --model "openai/whisper-large-v3"
```

동작:

1. `data/benchmarks/audios/mp3` 아래의 데이터셋 디렉터리를 자동 탐색한다.
2. 모든 `(dataset, mp3 filename)` 작업을 GPU 수에 맞게 분할한다.
3. GPU마다 Whisper 모델 프로세스 하나를 생성한다.
4. 각 MP3를 30초 chunk로 나누고 batch inference를 수행한다.
5. 대응하는 `stt/{dataset}` 디렉터리에 JSON을 저장한다.

예시:

```text
Input
data/benchmarks/audios/mp3/implicitqa/01GfDtSMG4s.mp4.mp3

Output
data/benchmarks/audios/stt/implicitqa/01GfDtSMG4s.mp4.mp3.json
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

이미 대응하는 STT JSON이 존재하면 해당 MP3는 건너뛴다.

### Step 3. STT 노이즈 필터링

관련 파일:

```text
filter_stt_noise.py
```

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

현재 필터 기준:

1. `transcript` 필드가 없으면 filtered
2. 공백 제거 후 길이가 4자 미만이면 filtered
3. 정의된 환각 키워드가 포함되면 filtered
4. 허용 문자 비율이 50% 미만이면 filtered
5. 단어가 5개보다 많고 고유 단어 비율이 20% 미만이면 filtered

현재 환각 키워드:

```text
YTN
Copyright
Subtitles by
Amara.org
[Music]
(Music)
(Applause)
[Applause]
```

### Step 4. Audio-only LLM 추론

관련 파일:

```text
run_infer_mp.py
run_all_inference.sh
```

단일 데이터셋 및 모델 실행:

```bash
OMP_NUM_THREADS=1 python run_infer_mp.py \
  --gpus "0,1,2,3" \
  --dataset "implicitqa" \
  --model_version "qwen25_vl"
```

지원하도록 구현된 모델 이름:

```text
qwen25_vl
qwen3_vl
eagle25
```

추론 프롬프트는 다음 정보만 모델에 제공한다.

```text
Timestamped transcript
Question
Options
Required answer/evidence/reasoning format
```

이미지 또는 비디오 tensor는 모델에 전달하지 않는다.

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

전체 모델과 데이터셋 조합을 실행하려면 `run_all_inference.sh`의 `MODELS`, `DATASETS`, `GPUS` 목록을 먼저 수정한 뒤 실행한다.

```bash
chmod +x run_all_inference.sh
./run_all_inference.sh
```

로그는 다음 위치에 저장된다.

```text
logs/{model}_{dataset}_{timestamp}.log
```

## 6. 현재 코드 기준 실행 전 확인사항

이 절은 GitHub commit 전에 반드시 확인해야 하는 현재 구현 상태를 기록한다.

### 6.1 하드코딩 경로

다음 경로는 실행 환경에 맞게 수정해야 한다.

```text
extract_audio.py
  JSON_PATH

run_infer_mp.py
  JSON_PATH
  Hugging Face authentication
```

### 6.2 `filter_stt_noise.py`의 목적지 변수

현재 파일 처리 loop에서는 `dst_file`과 `filtered_file`을 사용하기 전에 정의해야 한다.

각 파일을 읽은 직후 다음 경로를 구성해야 한다.

```python
dst_file = os.path.join(dst_dataset_path, filename)
filtered_file = os.path.join(filtered_dataset_path, filename)
```

이 정의가 없으면 각 파일에서 `NameError`가 발생하고 clean/filtered 결과가 생성되지 않는다.

### 6.3 raw STT와 clean STT 선택

현재 `run_infer_mp.py`는 다음 경로를 읽는다.

```text
data/benchmarks/audios/stt/{dataset}
```

노이즈 필터링된 STT만 사용하려면 이를 다음 경로로 변경해야 한다.

```text
data/benchmarks/audios/stt_clean/{dataset}
```

논문 결과가 raw STT 기준인지 clean STT 기준인지 명시하고 동일한 설정으로 재현해야 한다.

### 6.4 데이터셋 이름 매핑

오디오 추출 단계는 JSON의 `db`를 소문자로 변환하여 디렉터리를 만든다.

```python
item["db"].strip().lower()
```

반면 현재 `run_infer_mp.py`는 고정된 `db_list`와 STT 디렉터리 목록을 정렬한 뒤 위치 기준으로 매핑한다. 이 방식은 디렉터리 추가 또는 이름 변경 시 잘못된 DB가 연결될 수 있으며, `ImplicitQA` 같은 신규 데이터셋을 자동 지원하지 않는다.

재현 코드에서는 다음과 같이 JSON의 `db`를 동일한 규칙으로 정규화해 직접 비교하는 방식을 권장한다.

```python
target_data = [
    item
    for item in all_data
    if item["db"].strip().lower() == args.dataset
]
```

`--dataset` 값은 실제 디렉터리 이름과 같아야 한다.

### 6.6 상대 경로와 실행 위치

`run_infer_mp.py`의 argparse `choices`는 프로그램 시작 시 STT 디렉터리를 읽는다. 따라서 다음 경로가 실행 전에 존재해야 한다.

```text
data/benchmarks/audios/stt
```

또한 스크립트를 다른 working directory에서 실행하면 상대 경로가 달라진다. 재현 시 실행 위치를 고정하거나 `pathlib.Path(__file__)` 기준 경로로 변경하는 것이 안전하다.


## 7. 빠른 실행 요약

```bash
cd Video-Oasis/src/preprocess/audio

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
