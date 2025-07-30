import argparse
import os
import matplotlib.pyplot as plt


from maze_generator import SHAPES

def count_keywords_in_files(directory):
    # Keywords to search for
    keywords = {
        "SOLVE: SUCCESS": 0,
        "SOLVE: FAIL": 0,
        "RECOGNIZE: SUCCESS": 0,
        "RECOGNIZE: FAIL": 0,
        "GENERATE: SUCCESS": 0,
        "GENERATE: FAIL": 0
    }

    # Function to search for keywords in a file
    def search_file(file_path):
        with open(file_path, 'r', errors='ignore') as file:
            for line in file:
                for keyword in keywords:
                    if keyword in line:
                        keywords[keyword] += 1

    # Walk through all files and directories
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                search_file(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Print the results
    for keyword, count in keywords.items():
        print(f"{keyword}: {count}")

    return keywords

# Function to calculate accuracy for each task
def calculate_accuracies(keywords):
    accuracies = {}
    for task in ["SOLVE", "RECOGNIZE", "GENERATE"]:
        successes = keywords.get(f"{task}: SUCCESS", 0)
        fails = keywords.get(f"{task}: FAIL", 0)
        total = successes + fails
        accuracies[task] = (successes / total) if total > 0 else 0
    return accuracies


def create_accuracy_chart(shapes, base_dir, save_path):
    all_accuracies = {}

    # Process each shape
    for shape in shapes:
        shape_dir = os.path.join(base_dir, shape)
        keywords = count_keywords_in_files(shape_dir)
        accuracies = calculate_accuracies(keywords)
        all_accuracies[shape] = accuracies

    # Create the bar chart
    tasks = ["SOLVE", "RECOGNIZE", "GENERATE"]
    colors = ['blue', 'green', 'red']
    x = range(len(shapes))
    
    fig, ax = plt.subplots()
    
    for i, task in enumerate(tasks):
        task_accuracies = [all_accuracies[shape][task] for shape in shapes]
        ax.bar([p + i * 0.2 for p in x], task_accuracies, width=0.2, label=task, color=colors[i])
    
    # Set chart labels and legend
    ax.set_xlabel("Shapes")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Task Accuracy for Each Shape ({save_path})")
    ax.set_xticks([p + 0.2 for p in x])
    ax.set_xticklabels(shapes)
    ax.set_ylim((0, 1))
    ax.legend()
    plt.savefig(save_path)
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description="Count specific keywords in files within a directory.")
    parser.add_argument("directory", type=str, help="Path to the top-level directory to search")
    parser.add_argument("save_path", type=str, help="Path to save the fig")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    shapes =  ["square", "cross", "spiral", "triangle", "C", "Z"]
    create_accuracy_chart(SHAPES, args.directory, args.save_path)