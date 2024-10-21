#!/usr/bin/env python3

import sys
import csv

def process_files(input_files, output_file):
    try:
        # Initialize a list to store the results
        results = []

        # Process each input file
        for input_file in input_files:
            with open(input_file, 'r') as file:
                lines = file.readlines()

            # Search for the specified text and the line following it
            for i in range(len(lines)):
                if "Evaluation scores for location Covington:" in lines[i]:
                    eval_name = lines[i].strip().split(": ")[1]
                    scores = lines[i + 1].strip().split(", ")

                    precision = scores[0].split(": ")[1]
                    recall = scores[1].split(": ")[1]
                    f1_score = scores[2].split(": ")[1]
                    cohen_kappa = scores[3].split(": ")[1]

                    # Print the matching line and the line following it
                    print(f"{eval_name}, {precision}, {recall}, {f1_score}, {cohen_kappa}")

                    # Add the lines to the results list
                    results.append([eval_name, precision, recall, f1_score, cohen_kappa])

        # Write the results to the CSV file
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            csvwriter.writerow(["Evaluation Name", "Precision", "Recall", "F1-Score", "Cohen Kappa"])
            # Write the data
            csvwriter.writerows(results)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./process_files.py <output_file> <input_file1> <input_file2> ...")
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    process_files(input_files, output_file)

# usage 
# ./process_files.py <output_file> <input_file1> <input_file2> ...