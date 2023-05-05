import csv, pickle

# Specify the input CSV file name
input_file = "cleaned_data.csv"

# Read the data from the CSV file
data = []
with open(input_file, "r", newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        data.append(tuple(row))

# Specify the output pickle file name
output_file = "cleaned_data.pkl"

# Save the data as a pickle file
with open(output_file, "wb") as file:
    pickle.dump(data, file)