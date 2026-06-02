# Audio Only Inference with Whisper
**Goal** : Visual 정보 없이 오디오 기반 STT 텍스트 정보만 가지고 LLM이 풀 수 있는 Question 필터링 

## 실행 환경
```
# requirements
transformers==4.57.0
accelerate
torch==2.4.1
torchaudio==2.4.1
torchvision==0.19.1
# Whisper/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 1. 디렉토리 구조
```
/data3/sjpark/workspace/LVU_release/Whisper
├── extracted_mp3/          # 비디오에서 추출한 오디오(.mp3/wav) 저장소
├── logs/                   # 실행 로그 저장소
├── modules/                # 핵심 기능 모듈 패키지
│   ├── __init__.py
│   ├── audio_utils.py      # 오디오 변환 및 처리 관련 함수
│   ├── data_loader.py      # Video-MME 데이터셋 로드 및 필터링
│   ├── model_wrapper.py    # LLM (Qwen2.5-VL 등) 모델 로드 및 추론 래퍼
│   └── stt_utils.py        # Whisper 모델 로드 및 STT 실행 로직
├── output/                 # 실험 결과(JSON) 저장소 : 추후 LLM모델별 추론 결과 분류 
│   ├── eagle25/            # Eagle2.5 모델 추론 결과 파일들 
│   ├── qwen25_vl/          # Qwen2.5-VL 모델 추론 결과 파일들 
│   └── qwen3_vl/          # Qwen3-VL 모델 추론 결과 파일들
├── STT/                    # Whisper로 생성된 자막(Transcript) JSON 파일
├── STT_clean/              # STT에서 noisy한 파일 필터링한 clean 파일 모음 
├── utils/                  # 보조 유틸리티 패키지
│   ├── __init__.py
│   └── prompt_builder.py   # STT 기반 LLM 프롬프트 생성(Construct) 로직
├── analysis_result_all.py  # 추론 결과(JSON) 통계 및 정확도 분석 스크립트
├── filter_stt_noise.py     # STT 폴더에서 noisy한 json 파일 필터링 수행 코드
├── main_log.out            # 전체 실행 로그 저장 
├── extract_audio.py        # 비디오 파일에서 오디오 추출 실행 스크립트
├── run_inference_mp.py     # LLM 추론 메인 실행 스크립트 (1 dataset, 1 llm model)
├── run_all_inference.sh    # 전체 추론 실행 쉘 스크립트 
├── run_stt.py              # Whisper STT 추출 메인 실행 스크립트
├── vqa_total.json          # 전체 QA DB                                                                
└── readme.md               # 프로젝트 설명 (소문자)
```
## 2. 파이프라인 & 코드 실행

### Step 1. 오디오 추출 
- 경로 설정 : JSON_PATH = "./Whisper/datasets/videomme_all.json" 

```python extract_audio.py```

### Step 2. STT 변환 (Whisper)
```python run_stt.py --batch_size 16 --gpu "0,1,2,3"```

### Step 2.5 STT 노이즈 필터링
- 추출된 STT에서 노이즈 포함 파일 필터링
- 필터링 로직
    - 길이 검사 : 너무 짧으면 삭제
    - Hallucination keyword test : '[music]', 'Please Subscribe~' 등의 환각 키워드 삭제
    - 영어, 한국어, 숫자, 기본 특수문자만 허용
    - 반복 패턴 검사 (같은 단어 무한 반복) : 고유 단어 비율이 20% 미만이면 삭제 

### Step 3. LLM 추론 (Qwen2.5/Qwen3/Eagle2.5)
- LLM이 QA 수행 후 타임스탬프로 근거 찾아서 말하게끔 설정하여 추후 정성적으로 확인할 수 있도록 설정 
- GPU 병렬 처리로 추론 시간 단축 
   ```bash
   # 1개 벤치마크, 1개 모델 기준 실행 코드
   OMP_NUM_THREADS=1 python run_inference_mp.py \
      --dataset [DATASET_NAME] \
      --model_version [MODEL_NAME] \
      --gpus [GPU_IDS]

   # parameters
   # --dataset : ['video-holmes', 'video-mme', 'tvbench', 'vcr-bench', 'rtv-bench', 'mvbench', 'longvideobench', 'mmrvbench', 'mlvu_test']
   # -- model_version : ['qwen25_vl','qwen3_vl','eagle25']

   # 전체 실행 (background)
   chmod +x run_all_inference.sh
   nohup ./run_all_inference.sh > main_log.out 2>&1 &
   ```

- 추론 결과는 ```./output/{model_name}/{dataset_name}``` 경로로, 추론 **모델별+벤치마크별**로 저장
   - ex. /data3/jylee/workspace/Whisper/output/qwen3_vl
   - predicted_option : 추론 결과
      - 프롬프트에서 "Don't GUESS" 로 명시했기 때문에, 모델이 정답 내리기 힘들다고 자체 판단한 경우 'Unknown' 출력 
   - Evidence : LLM이 판단한 정답의 근거(+timestamp)
```json
{
        "qid": "M5YKW6fhlss_2",
        "video": "M5YKW6fhlss.mp4",
        "db": "LongVideoBench",
        "question": "There are a few red photo frames hanging on a wooden wall. A man wearing black-rimmed glasses and a plaid shirt is explaining. To his right, there is a phone that is changing screens. In which scenes has this phone appeared before?",
        "options": [
            "A. Next to an insect wearing a pink coat",
            "B. Next to an insect with a tie",
            "C. In the hands of a woman wearing a black coat in the park",
            "D. In the hands of a man wearing blue pants in the park",
            "E. In the hands of a character without a nose or mouth"
        ],
        "ground_truth": "D",
        "predicted_option": "Unknown",
        "is_correct": false,
        "Evidence": "None",
        "Reasoning": "Information missing, guessing based on context."
    },
    {
        "qid": "001-2",
        "video": "fFjv93ACGo8.mp4",
        "db": "Video-MME",
        "question": "What is the genre of this video?",
        "options": [
            "A. It is a news report that introduces the history behind Christmas decorations.",
            "B. It is a documentary on the evolution of Christmas holiday recipes.",
            "C. It is a travel vlog exploring Christmas markets around the world.",
            "D. It is a tutorial on DIY Christmas ornament crafting."
        ],
        "ground_truth": "A",
        "predicted_option": "A",
        "is_correct": true,   
        "Evidence": "[0.0 - 66.2] In New at 6, we're looking at the seasonal displays that are popping up everywhere these days. Holiday decorations. They've come a long way since they started. 13 News Now anchor Phillip Townsend looked at the history behind Christmas ornaments.",
        "Reasoning": "The transcript clearly states that the video is an introduction to the history behind Christmas decorations, making it a news report."
    },
```

### Step 4. 분석
- ```analyze_result_all.py``` : STT 전체 + random 성능 확인

   ```bash
   python analyze_result_all.py
   ```

- ```analyze_stt_clean.py``` : 노이즈 필터링 STT(STT_clean) + random 성능 확인

   ```bash
   python analyze_stt_clean.py
   ```

## 3. 추론 결과 
 
   **전체 QA 대상 accuracy**

   | Model             | longvideobench    | mlvu_test       | mmrvbench         | mvbench           | rtv-bench          | tvbench           | vcr-bench        | video-holmes      | video-mme          | ALL (Weighted Avg)   |
   |-------------------|-------------------|-----------------|-------------------|-------------------|--------------------|-------------------|------------------|-------------------|--------------------|----------------------|
   | Random (Baseline) | 21.33% (285/1337) | 16.67% (84/502) | 9.64% (121/1257)  | 29.72% (892/3000) | 30.39% (1400/4608) | 32.90% (726/2205) | 24.03% (123/511) | 16.67% (306/1837) | 25.00% (675/2700)  | 25.68% (4612/17957)  |
   | eagle25           | 14.06% (188/1337) | 7.57% (38/502)  | 13.37% (168/1257) | 18.80% (564/3000) | 11.09% (511/4608)  | 24.22% (534/2205) | 19.57% (100/511) | 21.45% (394/1837) | 38.93% (1051/2700) | 19.76% (3548/17957)  |
   | qwen25_vl         | 20.34% (272/1337) | 7.97% (40/502)  | 9.15% (115/1257)  | 21.73% (652/3000) | 14.69% (677/4608)  | 24.17% (533/2205) | 22.31% (114/511) | 17.47% (321/1837) | 38.41% (1037/2700) | 20.94% (3761/17957)  |
   | qwen3_vl          | 16.68% (223/1337) | 8.57% (43/502)  | 10.58% (133/1257) | 20.77% (623/3000) | 17.32% (798/4608)  | 27.53% (607/2205) | 21.33% (109/511) | 17.26% (317/1837) | 42.48% (1147/2700) | 22.28% (4000/17957)  |
  

   **노이즈 필터링한 STT_clean에서만 accuracy 계산 + 랜덤 성능 포함**

   |       Model       |       longvideobench        |         mlvu_test          |         mmrvbench          |            mvbench            |          rtv-bench           |            tvbench            |         vcr-bench          |         video-holmes         |           video-mme           |
   |-------------------|-----------------------------|----------------------------|----------------------------|-------------------------------|------------------------------|-------------------------------|----------------------------|------------------------------|-------------------------------|
   | Random (Baseline) | 20.93% (37/175) [Miss:1162] | 16.67% (28/170) [Miss:332] | 9.47% (49/522) [Miss:735]  | 28.89% (355/1229) [Miss:1771] | 30.67% (183/598) [Miss:4010] | 35.66% (413/1158) [Miss:1047] | 25.00% (46/185) [Miss:326] | 16.67% (178/1068) [Miss:769] | 25.00% (392/1569) [Miss:1131] |
   |      eagle25      | 12.57% (22/175) [Miss:1162] | 17.06% (29/170) [Miss:332] | 14.18% (74/522) [Miss:735] | 31.00% (381/1229) [Miss:1771] | 30.10% (180/598) [Miss:4010] | 36.79% (426/1158) [Miss:1047] | 37.84% (70/185) [Miss:326] | 25.84% (276/1068) [Miss:769] | 55.70% (874/1569) [Miss:1131] |
   |     qwen25_vl     | 24.00% (42/175) [Miss:1162] | 15.88% (27/170) [Miss:332] | 9.96% (52/522) [Miss:735]  | 23.43% (288/1229) [Miss:1771] | 28.09% (168/598) [Miss:4010] | 28.41% (329/1158) [Miss:1047] | 35.14% (65/185) [Miss:326] | 21.63% (231/1068) [Miss:769] | 54.49% (855/1569) [Miss:1131] |
   |     qwen3_vl      | 16.00% (28/175) [Miss:1162] | 20.00% (34/170) [Miss:332] | 12.64% (66/522) [Miss:735] | 31.98% (393/1229) [Miss:1771] | 34.11% (204/598) [Miss:4010] | 38.34% (444/1158) [Miss:1047] | 35.68% (66/185) [Miss:326] | 22.10% (236/1068) [Miss:769] | 59.78% (938/1569) [Miss:1131] |