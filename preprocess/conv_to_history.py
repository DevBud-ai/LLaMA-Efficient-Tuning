
import argparse
import json
import pathlib

def main(args):
    data_path = pathlib.Path(args.data_path)
    with data_path.open() as f:
        data = json.load(f)
    
    new_data = []
    
    for item in data:

        instruction = ''
        output = ''
        history = []
        
        for i in range(len(item['conversations']) // 2):
            history.append([item['conversations'][2*i]['value'], item['conversations'][2*i+1]['value']])
        
        if len(history) < 1:
            continue

        new_data.append({
            "instruction": history[-1][0],
            "output": history[-1][1],
            "history": history[:-1]
        })
    json.dump(new_data, open(args.output_path, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="flow-gpt.json")
    parser.add_argument(
        "--output_path", type=str, default="flowgpt-data-conversation.json"
    )
    args = parser.parse_args()
    main(args)