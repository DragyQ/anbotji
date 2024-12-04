from dotenv import load_dotenv
import os
import google.generativeai as genai

# load api_key from env file
load_dotenv()
API_KEY = os.getenv('gemini_api_key')

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Returns the concatenated user response after the interaction is over
def chatbot(model, max_iterations = 5):
    # TODO: replace this with parser
    questions = [
        "What are your hobbies?",
        "How do you usually spend your weekends?",
        "What do you value most in a friendship?",
        "Can you describe your ideal vacation?"
    ]

    context = 'I am Anbotji, a AI chatbot that asks the user certain questions as well as responding to their previous answers to simulate a normal conversation'
    prompt_context = 'Connect the previous user response with the next question you are about to ask to the user in a natural way'
    initial_question = "Hello! I'm Anbotji! I'm here to have a nice chat with you and hopefully get to learn more about you! To start, what's your name?"
    end_of_conversation_context = "Connect the previous user response and then try to end the conversation, don't give room for them to respond"
    chat = model.start_chat(
        history = [
            {
                'role': 'model',
                'parts': context
            },
            {
                'role': 'model',
                'parts': initial_question
            }
        ]
    )

    user_responses = []
    for i in range(max_iterations + 1):
        if i == 0:
            print(f"\nAnbotji: {initial_question}\n")
        else:
            # TODO: select a random question (for now it's sequential)
            question = '' if i == max_iterations else questions[i - 1]

            # get the last user response
            last_user_resp = user_responses[-1]

            # create the input prompt with context
            prompt = f"{last_user_resp}\n\n{question}\n\n{end_of_conversation_context if i == max_iterations else prompt_context}"

            # generate the chatbot's response
            response = chat.send_message(prompt)
            print(f"\nAnbotji: {response.text}")

        if i == max_iterations:
            break 

        # get user response
        user_response = input("You: ")
        user_responses.append(user_response)

    # Concatenate all user responses into a single string
    final_user_response = " ".join(user_responses)
    return final_user_response

# Run the chatbot
if __name__ == "__main__":
    chatbot(model, 2)
