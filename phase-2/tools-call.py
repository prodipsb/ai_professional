# ====================================================
# import libraries
# ====================================================

from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq()


# ====================================================
# STEP 1: DEFINE THE ACTUAL PYTHON FUNCTION
# ====================================================

# This is a REAL function that does REAL work.
# The LLM will never see this code - it only sees a DESCRIPTION of it (below).

def check_experience_match(candidate_years, required_years):
    """
    Simple real logic: compares two numbers:
    This is deliberately basic so you can focus on the MECHANISM
    of function calling, not complex logic.
    """

    if candidate_years >= required_years:
        gap = candidate_years - required_years
        return {
            "meets_requirement": True,
            "message": f"Candidate exceeds requirement by {gap} years" if gap > 0 else "Candidate exactly meets requirement"
        }
    else:
        gap = required_years - candidate_years
        return {
            "meets_requirement": False,
            "message": f"Candidate is short by {gap} years"
        }

# ====================================================
# STEP 1: DEFINE THE ACTUAL PYTHON FUNCTION ENDED
# ====================================================



# =======================================================
# STEP 2: DESCRIBE THE FUNCTION TO THE LLM
# =======================================================

# The LLM can't read Python code. Instand, we describe the function
# using a structured format (JSON Schema) so it understands:
# - What the function is called
# - What is does (in plain English)
# - What arguments it needs, and their types


tools = [
    {
        "type": "function",
        "function": {
            "name": "check_experience_match",
            "description": "Checks whether a candidate's year of experience meets a job's required years of experience",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_years": {
                        "type": "number",
                        "description": "How many years of experience the candidate has"
                    },
                    "required_years": {
                        "type": "number",
                        "description": "How many years of experience the job requires"
                    }
                },
                "required": ["candidate_years", "required_years"]
            }
        
        }
    }
]

# =======================================================
# STEP 2: DESCRIBE THE FUNCTION TO THE LLM ENDED
# =======================================================




# ====================================================
# STEP 3: SEND THE MESSAGE + TOOLS TO THE LLM
# ====================================================

user_question = "The job requires 5 years of experience. I have 3 years. Am I qualified?"

messages = [
    {
        "role": "user",
        "content": user_question
    }
]


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,                # We're telling the LLM what tools are available
    tool_choice="auto"          # "auto" means: LLM decides IF it needs to use a tool
)

# ====================================================
# STEP 3: SEND THE MESSAGE + TOOLS TO THE LLM ENDED
# ====================================================


# ====================================================
# STEP 4: CHECK IF THE LLM WANTS TO CALL A FUNCTION
# ====================================================

response_message = response.choices[0].message

print("DEBUG - Did the LLM request a tool call?")
print(response_message.tool_calls)
print()


if response_message.tool_calls:

    # The LLM decided it needs to call a function.
    # tool_calls is a LIST because the LLM could technically request
    # multiple function calls at once (not today, but good to know)

    tool_call = response_message.tool_calls[0]

    function_name = tool_call.function.name

    # arguments come back as a JSON STRING, so we parse it into a dict

    fucntion_args = json.loads(tool_call.function.arguments)

    print(f"LLM wants to call: {function_name}")
    print(f"With arguments: {fucntion_args}")
    print()

    # ====================================================
    # STEP 4: ENDED
    # ====================================================



    # ====================================================
    # STEP 5: ACTUALLY RUN THE REAL PYTHON FUNCTION
    # ====================================================

    # This is the ONLY place real logic executes.
    # We match the function NAME the LLM requested to our REAL function,
    # then call it with the arguments the LLM extracted.

    if function_name == "check_experience_match":
        result = check_experience_match(**fucntion_args)

    print(f"Function result: {result}")
    print()

    # ====================================================
    # STEP 5: ACTUALLY RUN THE REAL PYTHON FUNCTION ENDED
    # ====================================================



    # ====================================================
    # STEP 6: SEND THE RESULT BACK TO THE LLM
    # ====================================================

    # We add TWO new messages to the conversation:
    # 1. The LLM's own request to call the tool (so it "remembers" asking)
    # 2. The actual result of running that tool

    messages.append(response_message)           # the LLM's tool-call request
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)           # convert our result back into a string
    })

    # Now ask the LLM to give a Final answer, not that it has real data
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )

    print("FINAL ANSWER FROM LLM:")
    print(final_response.choices[0].message.content)

else:
    # The LLM answered directly without needing a tool
    print("LLM answered directly:")
    print(response_message.content)


