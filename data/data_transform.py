import pandas
import json

csv_file_path = './data/raw_dev.csv'
json_save_file_path = './data/dev_data.json'

unnecessary_labels = ['article_id', 'speaker_id', 'speaker_number', 'split', 'essay_id']
raw_data = pandas.read_csv(csv_file_path)
data = raw_data.to_dict(orient='records')
transformed_data = list()

unknown_mask = '<unk>'

for entry in data:
    unknown_counter = 0
    transformed_entry = {'id': -1, 'text': '', 'labels': dict()}
    for key in entry.keys():
        # Handle missing or invalid data
        if entry[key] == 'unknown' or entry[key] == unknown_mask:
            entry[key] = unknown_mask
            unknown_counter += 1

        # Retain numeric values for certain labels
        if key in ['gender', 'race', 'education']:
            try:
                entry[key] = float(entry[key]) if entry[key] != unknown_mask else 0.0
            except ValueError:
                entry[key] = 0.0

        # Populate transformed data
        if key == 'conversation_id':
            transformed_entry['id'] = entry[key]
        elif key == 'essay':
            transformed_entry['text'] = entry[key]
        elif key not in unnecessary_labels:
            transformed_entry['labels'][key] = entry[key]

    # Only append entry if it doesn't have a lot of <unk> values (useless data)
    if unknown_counter < 10:
        transformed_data.append(transformed_entry)

# Save the transformed data to JSON
with open(json_save_file_path, mode='w', encoding='utf-8') as file:
    json.dump(transformed_data, file, indent=2)  # Added indentation for pretty printing
