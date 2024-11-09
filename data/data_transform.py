import pandas
import json

# define the constant mapping from int to strings
gender_map = {'1': 'Male', '2': 'Female'}
race_map = {
    '1': 'White',
    '2': 'Hispanic / Latino',
    '3': 'Black / African American',
    '4': 'Native American / American Indian',
    '5': 'Asian / Pacific Islander'
}
education_map = {
    '1': 'Less than a high school diploma',
    '2': 'High school degree or diploma',
    '3': 'Technical / Vocational School',
    '4': 'Some college',
    '5': 'Two year associate degree',
    '6': 'College or university degree',
    '7': 'Postgraduate / professional degree'
}
unknown_mask = '<unk>'

csv_file_path = './data/raw_data.csv'
json_save_file_path = './data/data.json'

unnecessary_labels = ['article_id', 'speaker_id', 'speaker_number', 'split', 'essay_id']
raw_data = pandas.read_csv(csv_file_path)
data = raw_data.to_dict(orient = 'records')
transformed_data = list()
for entry in data:
    unknown_counter = 0
    transformed_entry = {'id': -1, 'text': '', 'labels': dict()}
    for key in entry.keys():
        # map strings to int values for certain labels
        if key == 'gender':
            entry[key] = gender_map.get(entry[key], '<unk>')
        elif key == 'education':
            entry[key] = education_map.get(entry[key], '<unk>')
        elif key == 'race':
            entry[key] = race_map.get(entry[key], '<unk>')

        # replace unknown values with <unk> mask
        if entry[key] == 'unknown' or entry[key] == unknown_mask:
            entry[key] = unknown_mask
            unknown_counter += 1

        if key == 'conversation_id':
            transformed_entry['id'] = entry[key]
        elif key == 'essay':
            transformed_entry['text'] = entry[key]
        elif key not in unnecessary_labels:
            transformed_entry['labels'][key] = entry[key]
            
    # only append entry if it doesn't have a lot of <unk> values (useless data)
    if unknown_counter < 10:
        transformed_data.append(transformed_entry)
        
with open(json_save_file_path, mode = 'w', encoding = 'utf-8') as file:
    json.dump(transformed_data, file, indent = 2) # added indentation for pretty printing
