import numpy as np
import os


def get_all_file(participants, data_dir, trial_names=None, to_include=(), to_exclude=(), file_ext=None):
    all_path = []
    parts = []
    if trial_names and len(trial_names) != len(participants):
        trial_names = [trial_names for _ in participants]

    for part in participants:
        add_session = False
        all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        sessions = [name for name in all_files if "ession" in name]
        if len(sessions) != 0:
            all_files = os.listdir(f"{data_dir}{os.sep}{part}/{sessions[0]}")
            add_session = sessions[0]
        if len(all_files) == 0:
            print("No files found for participant")
            continue
        if len(to_include) != 0:
            all_files = [file for file in all_files if any(ext in file for ext in to_include)]
        if len(to_exclude) != 0:
            all_files = [file for file in all_files if not any(ext in file for ext in to_exclude)]
        if file_ext:
            all_files = [file for file in all_files if file.endswith(file_ext)]
        if add_session:
            all_files = [f"{data_dir}{os.sep}{part}{os.sep}{add_session}{os.sep}{file}" for file in all_files
                         ]
        else:
            all_files = [
                f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files
            ]  # if "gear" in file and "less" not in file and "more" not in file and "result" not in file]

        final_files = all_files if not trial_names else []
        parts.append([part for _ in final_files])
        all_path.append(final_files)
    return sum(all_path, []), sum(parts, [])
