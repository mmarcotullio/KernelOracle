input_file_path_name = "out_ab_nginx"

with open(f"./data/{input_file_path_name}.txt") as data_file:
    lines = data_file.readlines()

with open(f"./data/scheduling_data_{input_file_path_name}.csv", "w") as out:
    out.write("task_code,time,name,pid\n")

    for line in lines:
        if "=>" not in line:
            continue

        tokens = line.split()

        # code is first token like *A0 or *.
        code = tokens[0].lstrip("*")

        # find time value: the number right before the word "secs"
        if "secs" not in tokens:
            # skip weird lines, or raise if you prefer
            continue

        secs_index = tokens.index("secs")
        if secs_index == 0:
            continue  # malformed

        time_secs = tokens[secs_index - 1]

        # name/pid is after => (usually last token)
        name_pid = tokens[-1]
        if ":" in name_pid:
            name, pid = name_pid.split(":", 1)
        else:
            name, pid = name_pid, ""

        out.write(f"{code},{time_secs},{name},{pid}\n")
