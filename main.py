from dotenv import load_dotenv
from questions_parser import PersonalityAssessmentParser
from random import choice
from typing import Tuple, Dict
from helper import get_input, race_map, gender_map, education_map, PERSONALITY_TRAITS, IRI_TRAITS
from model import predict_outputs

import os
import google.generativeai as genai


# load api_key from env file
load_dotenv()
API_KEY = os.getenv('gemini_api_key')

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="You are an interactive bot named Anbotji. Users will converse with you, while"
    "you will be given a list of questions to ask them. Ask them one question at a time."
)

# Get random questions to ask user.
parser = PersonalityAssessmentParser("questions.json")

def get_questions():
    questions = []
    
    for test_type in parser.get_test_types():
        choices = parser.get_questions_by_test_type(test_type)
        q = choice(choices)
        
        # Reselect question if we already chose it before for a diff
        # test type.
        while q in questions and len(choices) != 0:
            choices.remove(q)
            q = choice(choices)
        
        questions.append(q.get('question', ''))
    
    return questions


# Method to help get all context model will need about the user.
def get_context() -> Tuple[str, Dict[str, str], Dict[str, float]]:
    """
    Returns user's name and two maps: one containing demographic info,
    and the other containing the user's empathy and distress level.
    """
    name = get_input("\nEnter your name: ", r"\w+\s?\w*")
    gender = get_input(
        "\nEnter your gender:\n" +
        "\n".join(f"{k}: {v}" for k,v in gender_map.items()) + "\n",
        r"\d{1}"
    )
    race = get_input(
        "\nEnter your race:\n" +
        "\n".join(f"{k}: {v}" for k,v in race_map.items()) + "\n",
        r"\d{1}"
    )
    education = get_input(
        "\nEnter your education level:\n" +
        "\n".join(f"{k}: {v}" for k,v in education_map.items()) + "\n",
        r"\d{1}"
    )
    age = get_input("\nEnter your age: ", r"\d{1,3}")
    income = get_input("\nEnter your yearly income: ", r"\d+")

    empathy = get_input(
        "\nHow empathetic are you on a scale of 1-7?\n"
        "1 being not empathetic at all, 7 being extremely empathetic.\n"
        "You can rate this as an integer, or float number: ",
        r"\d\.?\d*"
    )
    distressed = get_input(
        "\nHow distressed are you on a scale of 1-7?\n"
        "1 being not distressed at all, 7 being extremely distressed.\n"
        "You can rate this as an integer, or float number: ",
        r"\d\.?\d*"
    )
    
    # Organizing into dictionaries
    demographic_info = {
        "gender": gender,
        "race": race,
        "education": education,
        "age": age,
        "income": income
    }
    empathy_distress_info = {
        "empathy": empathy,
        "distress": distressed
    }
    
    return name, demographic_info, empathy_distress_info

# Returns the concatenated user response after the interaction is over
def chatbot(model, max_iterations = 9):
    print(
        "Hello, I am Anbotji! Nice to meet you : )\n"
        "Let's start our conversation by learning more about you!"
    )
    name, demographics, empathy_distress = get_context()
    context_demographics = {}
    context_demographics['gender'] = gender_map[int(demographics['gender'])]
    context_demographics['race'] = race_map[int(demographics['race'])]
    context_demographics['education'] = education_map[int(demographics['education'])]
    
    user_context = " ".join(
        [name, str(context_demographics), str(empathy_distress)]
    )
    
    questions = get_questions()
    context = (
        "I am Anbotji, a AI chatbot that asks the user certain questions as well as"
        "responding to their previous answers to simulate a normal conversation. "
        "The user's info such as their name and demographics will be provided to me."
    )
    prompt_context = 'Connect the previous user response with the next question you are about to ask to the user in a natural way'
    end_of_conversation_context = "Connect the previous user response and then try to end the conversation, don't give room for them to respond"
    chat = model.start_chat(
        history = [
            {
                'role': 'model',
                'parts': context
            },
            {
                'role': 'user',
                'parts': user_context
            }
        ]
    )

    # Ask all questions
    user_responses = [user_context]
    for i in range(max_iterations):
        question = choice(questions)
        questions.remove(question)

        # get the last user response
        last_user_resp = user_responses[-1]

        # create the input prompt with context
        prompt = f"{last_user_resp}\n\n{question}\n\n{prompt_context}"

        # generate the chatbot's response
        response = chat.send_message(prompt)
        print(f"\nAnbotji: {response.text}")

        # get user response
        user_response = input("You: ")
        user_responses.append(user_response)

    # Concatenate all user responses into a single string
    final_user_response = " ".join(user_responses[1:])
    
    # Say bye to user based on their last response
    end_of_conversation_prompt = f"{user_responses[-1]}\n\n{end_of_conversation_context}"
    print(f"\nAnbotji: {chat.send_message(end_of_conversation_prompt).text}")
    
    # Predict output
    personality, iri = predict_outputs(final_user_response, demographics, empathy_distress)
    
    for trait, output in zip(PERSONALITY_TRAITS, personality.flatten()):
        print(f"{trait}: {round(output.item(), 3)}")
    
    for trait, output in zip(IRI_TRAITS, iri.flatten()):
        print(f"{trait}: {round(output.item(), 3)}")
    

# Run the chatbot
if __name__ == "__main__":
    chatbot(model, 9)
