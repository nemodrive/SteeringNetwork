from argparse import ArgumentParser
import os.path
import re

"""
Generates configurators. Combination of all parameters that have variations.
Match pattern: 
    ${<type_of_function>, <parameters_comma_seperated>....}
    
E.g. 
    seed: ${list, 1,2,3}
    seed: ${eval, np.random.rand(10)}
"""


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def list_of_param(args_string):
    a = args_string.strip().split(",")
    return a


PATTERNS = {
    "list": list_of_param,
    "eval": eval
}


def get_method(pattern):
    method = re.findall("[^,]*", pattern)[0]
    args_str = pattern.replace(f"{method},", "")
    return PATTERNS[method](args_str)


def main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument('name', type=str, help='Name for experiment folder')
    arg_parser.add_argument('path', type=str, help='Base file path to generate configs from')
    arg_parser.add_argument('--override', action='store_true',
                            help='override out path folder')

    args = arg_parser.parse_args()

    base_file_path = args.path
    with open(base_file_path, "r") as myfile:
        cfg = myfile.readlines()

    for line_idx in range(len(cfg)):
        line = cfg[line_idx]
        cfg[line_idx] = []

        mtches = re.findall("\$\{(.*?)\}", line)
        if len(mtches) == 0:
            cfg[line_idx].append(line)
            continue

        lines = [line]
        for pattern in mtches:
            # Determine function to apply
            parameters = get_method(pattern)

            new_lines = []
            for ln in lines:
                for aux in parameters:
                    new_line = ln.replace(f"${{{pattern}}}", str(aux))
                    new_lines.append(new_line)
            lines = new_lines

        cfg[line_idx] = lines

    new_cfgs = [[x] for x in cfg[0]]

    for i in range(1, len(cfg)):
        next_cfgs = []
        for n_cfg in new_cfgs:
            for line in cfg[i]:
                next_cfgs.append(n_cfg + [line])
        new_cfgs = next_cfgs

    # Generate files
    dir_name = os.path.dirname(base_file_path)
    folder_name = os.path.join(dir_name, args.name)
    if os.path.exists(folder_name):
        print(f"Folder exists {folder_name}!!!")
        if args.override:
            print(f"FOLDER OVERRIDE!")
            os.removedirs(folder_name)
        else:
            print("ABORT MISSION!")
            exit(1)

    print(f"Generate out folder: {folder_name}")
    os.mkdir(folder_name)

    save_path = os.path.join(folder_name, os.path.basename(base_file_path))

    for i, file_lines in enumerate(new_cfgs):
        new_file_name = save_path.replace(".yaml", f"_{i}.yaml")

        with open(new_file_name, "w") as myfile:
            myfile.writelines(file_lines)


if __name__ == "__main__":
    main()
