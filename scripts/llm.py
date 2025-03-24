# from langchain_ollama import OllamaLLM
# from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# model = OllamaLLM(model="deepseek-r1", temperature=0.2, num_predict=4096)
# modelOpenAI = OpenAI(api_key="", model="")

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

_ = load_dotenv()


llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("gemini_api_key_2"), # type: ignore
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)




test_case_prompt = PromptTemplate(
    input_variables=["user_story", "jira_id", "acceptance_criteria"],
    template="""
### Task: AI Test Case Generator

#### **Objective**  
You are an AI test case generator. Your job is to analyze JIRA user stories and create **detailed, structured, and exhaustive** test scenarios and test cases.

---
#### **Instructions**  
1. **Extract key details** from the user story.
2. **Most Importantly you should generate the same output for the same input which means that the output provided by you should not change or vary not even in number if I provide the same user story multiple times as input.
3. **Derive acceptance criteria** from the provided input if not already provided, depending on the user, sometimes the user will provide acceptance criteria and sometimes user will not provide the acceptance criteria.
4. **Identify all possible test scenarios**, covering positive, negative, and edge cases, the coverage of the generated output should be atleast 95%.
5. **Generate test cases** for each scenario, following the given format.
6. **Ensure strict format consistency** to maintain readability and usability.
7. **Use the provided example** to understand the expected output structure.
8. **The response should strictly contain minimum 11-15 test scenarios.** 


---
#### **Example Output Format**
(Use this exact format in your response)

---
### **User Story**  
**Story Title:** [Extracted from JIRA]  
**Description:** [Extracted from JIRA]  
**JIRA Issue ID:** {jira_id}  

### **Acceptance Criteria**  
{acceptance_criteria}

---
### **Test Scenarios & Test Cases**  

#### **Test Scenario ID: TS_01**  
**Test Scenario:** [Describe the purpose of testing this scenario and start the sentence with "validate whether"]  

##### **Test Case ID: TC_01**  
- **Test Case:** [Describe the purpose of this test case and start the sentence with "validate whether"]  
- **Preconditions:** [Any necessary setup before execution]  
- **Test Data:** [Example test data if applicable, with the disclaimer "The test data is just for guidance and the actual test data is to be determined by the user."]  
- **Test Execution Steps:**  
  1. Step 1  
  2. Step 2  
  3. Step 3  
  4. Step 4
  5. Provide as many steps as possible by carefully analysing the test case.
- **Expected Outcome:** [Define the expected results]  
- **Pass/Fail Criteria:**  
  - **Pass:** [Conditions under which the test case passes]  
  - **Fail:** [Conditions under which the test case fails]  
- **Priority:** [Low | Medium | High]  
- **References:** {jira_id}

---
#### **Now generate test scenarios and test cases for the following user story:**

**User Story:**  
{user_story}  

**JIRA Issue ID:** {jira_id}  

**Expected Acceptance Criteria:**  
{acceptance_criteria}  

---
### **Response Format Requirements**  
1. **Maintain structure exactly as shown above.**  
2. **Include minimum 11-15 test scenarios covering different cases depending on the user story (Provide more test scenarios and test cases according to the requirement)**  
3. **Generate minimum 2-3 test cases per scenario depending on the test scenario (number of test cases may vary based on the type of scenario)**  
4. **Strictly follow the example format for readability and consistency.**  
5. **Avoid unnecessary explanations—output should be directly usable by QA engineers.**  
6. **The Provided test scenarios and test cases should not be limited to the provided number and if necessary, generate more test scenarios and test cases according to the provided user story.**
7. **The provided test scenarios and test cases should be in detailed and should cover even those test scenarios and test cases which might be missed even by an experienced QA Engineer.**



Now generate the response.
"""
)


user_story = "As an admin, I should be able to manage the assessments"
jira_id = "RD-409"
acceptance_criteria = """
1.The side panel should have an Assessment Hub section with Assessments and Question Bank tabs.
2.Assessments should be displayed as cards with title, type, tags, last modified details, and status indicators (Blue: Assigned, Gray: Unassigned).
3.A filter option should allow filtering by type (Self, Peer) and status (assigned, unassigned) with a count badge and an empty state message if no results match.
4.A search bar should enable searching by assessment title, updating the assessment count dynamically.
5.A three-dot menu should provide options to Duplicate (creates a copy) and Delete (only unassigned assessments, with confirmation).
"""


formatted_prompt = test_case_prompt.format(
    user_story=user_story,
    jira_id=jira_id,
    acceptance_criteria=acceptance_criteria
)

def find_next_id():
  files = os.listdir("outputs")
  max_id = -1
  for x in files:
    max_id = max(max_id,(int(x.split("output")[1].split(".")[0])))
  return max_id + 1

try:
  file_name = find_next_id()
except:
  file_name = 1
with open(f"outputs/{jira_id}_output{str(file_name)}.md", "w") as file:
    response = llm.invoke(formatted_prompt)  
    file.write("\n User Prompt :" + "  " + test_case_prompt.template + "\n")
    file.write("-------------------")
    file.write("------------------------- LLM output -------------------------------------------------- \n\n ")
    file.write(response.content + "\n") # type: ignore
    file.write("Tokens Outputed: " + str(len(response.content.split(" "))) + "\n") # type: ignore
    file.write("\n")
    file.write("End of Iteration" + "\n")
    file.write("---------------------------------------------------------------------------")
    file.write("\n")
  

    
    
    

    
