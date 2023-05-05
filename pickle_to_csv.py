import csv, pickle

with open('data.pkl', 'rb') as file:
        data = pickle.load(file)

# Specify the output file name
output_file = "data.csv"

# Write the data to a CSV file
with open(output_file, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in data:
        csv_writer.writerow(row)