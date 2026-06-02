import os

def rename_files():
    # 현재 경로 (.) 혹은 작업하려는 절대 경로 입력
    # 쉘에서 해당 폴더(videos/rtv-bench/videos)에 위치해 있다면 '.' 그대로 두시면 됩니다.
    target_dir = "/mnt/users/gtlim/workspace/src/benchmark/videos/rtv-bench/videos" 
    
    # 안전 장치: True일 때는 실제로 바꾸지 않고 출력만 합니다. 
    # 출력이 의도한 대로 나오면 False로 변경해서 실행하세요.
    dry_run = False 

    files = os.listdir(target_dir)
    count = 0

    print(f"--- 작업 시작 (Dry Run: {dry_run}) ---")

    for filename in files:
        # mp4 파일만 대상으로 함
        if not filename.endswith(".mp4"):
            continue

        # 로직: 파일 이름에서 '첫 번째 대문자'가 나오는 인덱스를 찾음
        first_upper_index = -1
        for i, char in enumerate(filename):
            if char.isupper():
                first_upper_index = i
                break
        
        # 대문자가 있고, 파일 이름의 첫 글자가 대문자가 아닌 경우(즉, 앞에 접두사가 있는 경우)
        if first_upper_index > 0:
            new_filename = filename[first_upper_index:]
            
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_filename)

            if dry_run:
                print(f"[예상 변경] {filename} -> {new_filename}")
            else:
                os.rename(old_path, new_path)
                print(f"[변경 완료] {filename} -> {new_filename}")
            
            count += 1

    if count == 0:
        print("변경할 파일이 없거나 패턴에 맞는 파일이 없습니다.")
    else:
        print(f"--- 총 {count}개의 파일 처리 완료 ---")

if __name__ == "__main__":
    rename_files()