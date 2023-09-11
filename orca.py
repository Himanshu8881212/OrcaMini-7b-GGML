# Define ANSI escape codes for color formatting
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
GREEN = '\033[92m'
ENDC = '\033[0m'

# Header box
print(f"{BLUE}***********************************************{ENDC}")
print(f"{GREEN}*              Orca - the assistant          *{ENDC}")
print(f"{BLUE}***********************************************{ENDC}")
print("\n\n")  # Add two empty lines

from ctransformers import AutoModelForCausalLM
from langchain.memory import ConversationBufferMemory

# Initialize chat history memory
memory = ConversationBufferMemory(memory_key="chat_history")

def load_llm():
    llm = AutoModelForCausalLM.from_pretrained("path\to\the\model\",
                                               model_type='llama',
                                               max_new_tokens=1096,
                                               repetition_penalty=1.13,
                                               temperature=0.1)
    return llm

def llm_function(message):
    # Add user's message to memory
    memory.chat_memory.add_user_message(message)
    
    # Get chat history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    system_prompt = f"### System:\nYou are an AI assistant. Your name is Orca who follows instructions extremely well. Help as much as you can.\n\n{chat_history}"
    prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
    
    llm = load_llm()
    response = llm(prompt)
    output_texts = response

    # Add AI's response to memory
    memory.chat_memory.add_ai_message(output_texts)

    return output_texts

# Continuous user input loop
while True:
    message = input(f"{RED}You:{ENDC} ")  # Red color for "You: "
    if message.lower() == 'bye':
        break
    response = llm_function(message)
    print(f"{YELLOW}Orca:{ENDC} {response}")  # Yellow color for "Orca: "
    print()  # Add an empty line after each message
