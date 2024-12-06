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


MODEL_OUTPUT_TO_TEXT = {
    "Conscientiousness": {
        "high":     "You are highly conscientious, showing great attention to detail, organization, and reliability.",
        "moderate": "You are reasonably conscientious, balancing responsibilities with flexibility.",
        "low":      "You may be more relaxed or spontaneous, with a lower focus on structure and organization."
    },
    "Openness": {
        "high":     "You are highly open-minded and curious, always eager to explore new ideas and experiences.",
        "moderate": "You are open to new experiences but also enjoy familiar, stable situations.",
        "low":      "You might prefer routine and familiarity, with less interest in abstract ideas or unconventional experiences."
    },
    "Extraversion": {
        "high":     "You are extroverted, outgoing, and energized by social interactions.",
        "moderate": "You have a balanced approach, enjoying social interactions but also valuing time alone.",
        "low":      "You are introverted, finding energy in solitude and preferring quiet settings over large social gatherings."
    },
    "Agreeableness": {
        "high":     "You are very agreeable, empathetic, and cooperative, always striving to maintain harmonious relationships.",
        "moderate": "You are generally cooperative, but may assert your opinions when necessary.",
        "low":      "You may be more assertive or competitive, often preferring direct confrontation or independence."
    },
    "Stability": {
        "high":     "You are emotionally stable, handling stress and challenges with calm and resilience.",
        "moderate": "You may experience occasional stress or anxiety, but you generally manage it well.",
        "low":      "You may find yourself prone to stress or emotional fluctuations in challenging situations."
    },
    "Perspective Taking": {
        "high":     "You have strong empathy and are able to understand others' perspectives easily.",
        "moderate": "You are generally empathetic, but may sometimes struggle to see things from others' points of view.",
        "low":      "You may find it challenging to empathize with others or see things from their perspective."
    },
    "Personal Distress": {
        "high":     "You may feel strongly distressed when witnessing others' struggles and may be deeply affected by their emotions.",
        "moderate": "You may feel some level of distress, but can usually manage or distance yourself from it.",
        "low":      "You may not feel deeply affected by others' emotions or struggles, staying calm in such situations."
    },
    "Fantasy": {
        "high":     "You have a vivid imagination and often immerse yourself in creative, fantastical thoughts.",
        "moderate": "You enjoy some imaginative thinking but are also grounded in reality.",
        "low":      "You are highly practical, preferring real-world logic and reasoning over imaginative or fantastical ideas."
    },
    "Empathetic Concern": {
        "high":     "You are highly compassionate, often feeling concern for others' well-being and offering help when needed.",
        "moderate": "You care about others' well-being, but may not always express it strongly.",
        "low":      "You may not express empathy or concern for others as strongly, focusing more on practical matters."
    }
}

def get_tipi_trait_ans(score: float, trait: str):
    if score <= 3:
        return MODEL_OUTPUT_TO_TEXT[trait]["low"]
    elif score <= 5:
        return MODEL_OUTPUT_TO_TEXT[trait]["moderate"]
    else:
        return MODEL_OUTPUT_TO_TEXT[trait]["high"] 

def get_iri_trait_ans(score: float, trait: str):
    if score <= 1:
        return MODEL_OUTPUT_TO_TEXT[trait]["low"]
    elif score <= 3:
        return MODEL_OUTPUT_TO_TEXT[trait]["moderate"]
    else:
        return MODEL_OUTPUT_TO_TEXT[trait]["high"]
    

if __name__ == "__main__":
    pass