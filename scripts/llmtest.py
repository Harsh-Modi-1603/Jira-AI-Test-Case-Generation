from langchain_ollama import OllamaLLM
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

_ = load_dotenv()

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("gemini_api_key_2"),
    model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=4096,
    timeout=None,
    max_retries=2,  # Keeping retries minimal to avoid excessive requests
)

# Define the prompt template for generating test cases
test_case_prompt = PromptTemplate(
    input_variables=["user_story", "jira_id", "acceptance_criteria", "batch_number"],
    template="""
### Task: AI Test Case Generator - Batch {batch_number}

#### **Objective**  
You are an AI test case generator. Your job is to analyze JIRA user stories and create **detailed, structured, and exhaustive** test scenarios and test cases.

---
### **User Story**  
**Story Title:** Extracted from JIRA  
**Description:** {user_story}  
**JIRA Issue ID:** {jira_id}  

### **Acceptance Criteria**  
{acceptance_criteria}

---
### **Test Scenarios & Test Cases**  
#### **Generate unique test cases in this batch. Ensure there is no duplication from previous batches.**  
1. Generate exactly 5 unique test scenarios in this batch.  
2. Each scenario should contain at least 2 detailed test cases.  
3. Follow the structured format as shown in the provided example.  
4. Ensure at least 95% coverage of the user story.  
5. Cover positive, negative, boundary, security, usability, and performance cases.  

---
**Now generate test scenarios and test cases for this batch:**
"""
)

# Function to generate test cases in multiple batches with rate limit handling
def generate_test_cases(user_story, jira_id, acceptance_criteria, total_scenarios=20, batch_size=5):
    num_batches = total_scenarios // batch_size
    all_responses = []

    for batch_number in range(1, num_batches + 1):
        formatted_prompt = test_case_prompt.format(
            user_story=user_story,
            jira_id=jira_id,
            acceptance_criteria=acceptance_criteria,
            batch_number=batch_number
        )

        try:
            response = llm.invoke(formatted_prompt)
            test_case_content = response.content
            all_responses.append(test_case_content)

            # Save to file
            with open("output.md", "a") as file:
                file.write(f"\n### Batch {batch_number} Generated Test Cases:\n")
                file.write(test_case_content)
                file.write("\n\n")

            print(f"Batch {batch_number} generated successfully.")

        except Exception as e:
            print(f"Error in Batch {batch_number}: {e}")
            continue  # Skip to the next batch

        # Respect the API rate limit by waiting 30 seconds before the next request
        print("Waiting for 30 seconds to comply with API rate limits...")
        time.sleep(30)

    return all_responses

# Example Input Data
user_story = "As a customer, I want to add my credit card details so that I can make purchases on the platform."
jira_id = "RD-301"
acceptance_criteria = """
1. Users should be able to enter their credit card number, expiration date, and CVV.
2. The system should validate the card details before saving them.
3. The system should not store CVV for security reasons.
4. Users should receive a confirmation upon successfully adding a card.
"""

# Generate test cases
generate_test_cases(user_story, jira_id, acceptance_criteria)