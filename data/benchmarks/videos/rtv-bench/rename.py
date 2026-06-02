import os

def rename_files():
    target_dir = "./videos" 
    dry_run = False 

    files = os.listdir(target_dir)
    count = 0

    print(f"--- Process started (Dry Run: {dry_run}) ---")

    for filename in files:
        if not filename.endswith(".mp4"):
            continue

        first_upper_index = -1
        for i, char in enumerate(filename):
            if char.isupper():
                first_upper_index = i
                break
        
        if first_upper_index > 0:
            new_filename = filename[first_upper_index:]
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_filename)
            
            if dry_run:
                print(f"[Preview] {filename} -> {new_filename}")
            else:
                os.rename(old_path, new_path)
                print(f"[Renamed] {filename} -> {new_filename}")
            count += 1
            
    if count == 0:
        print("No files to rename or no files match the pattern.")
    else:
        print(f"--- Total {count} files processed ---")

if __name__ == "__main__":
    rename_files()