import json
import random
from sklearn.model_selection import train_test_split

# Save data to file
def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 2)

# Split data into train, dev, and test sets, then saves them to 
def split_data(file_path, train_ratio=0.7, dev_ratio=0.2, test_ratio=0.1,
                train_save_path='train_data.json', dev_save_path='dev_data.json', test_save_path='test_data.json'):
    # load json data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)  # Shuffle the data

    # Split data
    train_data, remaining_data = train_test_split(data, test_size = (1 - train_ratio))
    dev_ratio_adjusted = dev_ratio / (dev_ratio + test_ratio)  # Adjust dev ratio relative to the remaining data
    dev_data, test_data = train_test_split(remaining_data, test_size = (1 - dev_ratio_adjusted))

    # save each data split to file
    save_to_json(train_data, train_save_path)
    save_to_json(dev_data, dev_save_path)
    save_to_json(test_data, test_save_path)

    return train_data, dev_data, test_data

# testing purposes
file_path = './data/data.json'
train_data, dev_data, test_data = split_data(file_path)
