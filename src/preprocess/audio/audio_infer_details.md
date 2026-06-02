# Pipelines
1. 오디오 추출 : mp4 -> mp3
    
2. By Whisper : mp3 -> STT 
    - 오디오 추출에 성공한 videos 대상으로 Whisper 통한 STT 생성

3. Noised STT 필터링 
    - 특정 문자 반복, 특수 문자 포함 등의 노이즈 가진 STT 파일 필터링 

3. LLM Inference
    - STT 변환된 텍스트 전체를 LLM에 입력해서 QA 수행 

# 오디오 추출 
인간의 목소리를 포착하기 위한 목적으로 오디오 추출
```python
import subprocess
command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", 
            "-codec:a", "libmp3lame", "-q:a", "4",
            output_path
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path
        except:
            return None
```
- 오디오 채널을 1개로 merge (stereo->mono)
- Sampling Rate : 16 kHz
- mp3 encoding : libmp31ame
- mp3 quality : 165kbps

## Details

- 대부분의 딥러닝 기반 음성 인식 모델(ex. Whisper)는 내부적으로 16kHz sampling rate와 Mone 채널 입력을 기본으로 함. -> 이후 추론 때 resampling 수요가 없어 추론 속도 향상
- 정보 추출 용도이기 때문에 적당한 quality로 선택해 파일 용량 관리


# STT 변환 

- 음성 인식 모델 : **openai/whisper-large-v3**
    - 현재 Whisper 모델 중 가장 성능이 뛰어난 최신 버전 

- STT json 구조
    - video_id : 6hBbXVkgxGE.mp4.mp3
    - transcript : 변환된 전체 텍스트 결과
    - timestamp ; 구간별 타임스탬프 정보 
    - STT json 저장 예시 
    ```json
    {
    "video_id": "6hBbXVkgxGE.mp4.mp3",
    "transcript": " Welcome back movie lovers. Today we're diving into the heart-pounding 2021 action thriller Wrath of Man, ... And as usual, see you on the next one.",
    "timestamps": [
        {
            "timestamp": [
                0.0,
                167.2
            ],
            "text": " Welcome back movie lovers. Today we're diving into the heart-pounding 2021 action thriller Wrath of Man, starring Hollywood heavyweight Jason Statham. The film follows ... is more to H, calling him a dark spirit."
        },
        {
            "timestamp": [
                167.58,
                171.56
            ],
            "text": " Later that night H goes into the house of a guard named Dana, but when he finds a stack"
        },
        {
            "timestamp": [
                171.56,
                173.76
            ],
            "text": " of money in her apartment, he interrogates her."
        }, 
        ...
    }
    ```
## Details

```python
class WhisperTranscriber:
    def __init__(self, model_id="openai/whisper-large-v3", device=0):
        
        . . .

    def transcribe(self, audio_path, batch_size=16):
        try:
            res = self.pipe(
                audio_path, 
                chunk_length_s=30, 
                batch_size=batch_size, # 16
                return_timestamps=True
            )
            return res
        except Exception as e:
            # 에러 발생 시에도 dict 형태 유지
            return {"text": f"Error: {str(e)}", "chunks": []}
```
** ```chunck_length_s = 30 & batch_size = 16```
- Whisper 모델은 한 번에 최대 30초 길이의 오디오만 한 번에 처리하도록 학습
    - 오디오를 30초씩 Slicing해서 모델에 입력  (as 30 sec chunks)
    - 각 chunk를 텍스트로 변환
    - 변환된 text chunk들을 다시 stitch하여 전체 텍스트 생성
- batch : chunk 16개를 GPU에 태워 한꺼번에 병렬 연산 

# Filtering STT noises

- **필터링 로직**
    -  길이 검사 (너무 짧으면 삭제) : text length < 4
    - 환각 키워드 검사 
    ```python
    HALLUCINATION_KEYWORDS = [
        "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", 
        "구독과 좋아요", 
        "YTN", "Copyright", 
        "자막 제작", "Subtitles by", "Amara.org", 
        "배경 음악", "[Music]", "(Music)", "음악", 
        "박수", "(Applause)", "[Applause]"
    ]
    ```
    - 특수문자 제외 : 영어, 한국어, 숫자, 기본 특수문자만 허용
        - 유효 문자 50% 미만 시 노이즈로 간주 
    - 반복 패턴 검사 (같은 단어 무한 반복)


# LLM Inference 

- 사용 모델 : Qwen2.5 VL, Qwen3 VL, Eagle 2.5 
- 프롬프트 구성
    - 모델의 사전 지식을 사용하지 않고, 오직 주어진 텍스트 안에서만 답을 찾도록 제한
    - Instructions
        - (1) 스크립트와 시간을 꼼꼼히 읽어라
        - (2) 선지 중에서 정답을 골라라
        - (3) **CRITICAL** 정답의 근거가 되는 타임스탬프 구간을 반드시 인용해라 (찍기 방지)
        - (4) **NO GUESSING** 정보가 없으면 불확실성을 인정하고 문맥상 가장 그럴듯한 것을 골라라
        - (5) "Answer: " 뒤에는 선지에 해당하는 알파벳만 적어라. 
    - 타임스탬프별 script
    - question
    - option 
    - Format : output 형식 지정 
        - Answer : 답
        - Evidence : 증거가 되는 타임스탬프의 텍스트, 없다면 "None"
        - Reasoning : 답을 고른 이유 설명 

```python
prompt = f"""You are an AI assistant tasked with answering questions based STRICTLY on the provided video transcript.

[INSTRUCTIONS]
1. Read the transcript with timestamps carefully.
2. Select the correct option ({valid_range}).
3. **CRITICAL**: You MUST cite the specific timestamp range (e.g., [12.5 - 15.0]) from the transcript that supports your answer.
4. **DO NOT GUESS**: If the provided transcript does not contain enough information to answer the question, acknowledge the uncertainty in the 'Reasoning' section, but still select the most plausible option based on context.
5. **OUTPUT FORMAT**: For the 'Answer:' field, write ONLY the option letter (e.g., 'Answer: A'). Do not add any other text or parentheses.
[Transcript]
{transcript_context}

[Question]
{question}

[Options]
{options_str}

[Format]
Answer: (Option Letter)
Evidence: (Quote the text and timestamp. If none, write "None")
Reasoning: (Explain why you chose this option. If you are guessing, explicitly state "Information missing, guessing based on context.")"""
    return prompt.strip()
```


