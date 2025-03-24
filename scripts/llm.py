from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


llm_model = ChatGoogleGenerativeAI(
    api_key=os.getenv("gemini_api_key_2"),
    model="gemma-3-27b-it",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


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
""",
)
