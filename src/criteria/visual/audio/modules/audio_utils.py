import os
import subprocess

class AudioProcessor:
    def __init__(self, temp_dir="data/benchmarks/audios/mp3"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_audio(self, task):

        video_path, dataset = task

        video_filename = os.path.basename(video_path)

        dataset_dir = os.path.join(self.temp_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        output_path = os.path.join(
        dataset_dir,
        f"{video_filename}.mp3",
        )

        if os.path.exists(output_path):
            return output_path

        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            output_path,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return output_path
        except Exception as e:
            print(f"Failed to extract {video_path}: {e}")
            return None