input_file_path_name = 'out_ab_nginx'

with open("./data/" + input_file_path_name + ".txt") as data_file:
    lines = data_file.readlines()

lookup_map = {}

with open("./data/scheduling_data_" + input_file_path_name + ".csv", mode="w") as schedulind_Data:
    schedulind_Data.write("task_code,time,name,pid\n")

    for line in lines:
        tokens = [token for token in line.split(" ") if token]

        if len(tokens) != 4 and len(tokens) != 6:
            print(tokens)
            raise ValueError("Length of row does not seem right! Stopping.")

        code = tokens[0].strip("*").strip(" ")

        if len(tokens) == 6:
            name = tokens[5].strip("\n").split(":")[0]
            pid = tokens[5].strip("\n").split(":")[1]
            lookup_map[code] = (name, pid)

        row = [code, tokens[1], lookup_map[code][0], lookup_map[code][1]]
        schedulind_Data.write(",".join(row) + "\n")
