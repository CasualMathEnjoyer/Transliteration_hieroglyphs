import os
import json


def count_lines_in_json(filepath):
    with open(filepath, 'r') as file:
        return sum(1 for line in file)


def main():
    folder_path = '/home/katka/Downloads/models_LSTM'
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        line_count = count_lines_in_json(file_path)
        print(f'{json_file}: {line_count} lines')


if __name__ == "__main__":
    main()
