import re


gender_map = {1:"Male", 2:"Female"}
race_map = {1:"White", 2:"Hispanic / Latino", 3:"Black / African American", 4:"Native American / American Indian",
            5: "Asian / Pacific Islander", 6: "Other"}
education_map = {1: "Less than a high school diploma",
                 2: "High school degree or diploma",
                 3: "Technical / Vocational School",
                 4: "Some college",
                 5: "Two year associate degree",
                 6: "College or university degree",
                 7: "Postgraduate / professional degree"}

PERSONALITY_TRAITS = [
    "Conscientiousness",
    "Openness",
    "Extraversion",
    "Agreeableness",
    "Stability"
]
IRI_TRAITS = [
    "Perspective Taking",
    "Personal Distress",
    "Fantasy",
    "Empathetic Concern"
]


def get_input(prompt: str, validation_re: str) -> any:
    user_input = input(prompt)
    valid_pattern = re.compile(validation_re)
    
    while not valid_pattern.match(user_input):
        print(f"Invalid input! Pleas re-enter val to match R.E.: {valid_pattern.pattern}")
        user_input = input(prompt)

    return user_input


if __name__ == "__main__":
    pass