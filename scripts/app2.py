from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize FastAPI app
app = FastAPI(title="Test Case Generator API", 
              description="API for generating test cases based on user stories")

# Load environment variables
load_dotenv()

# Initialize Google Generative AI model
def get_llm():
    return ChatGoogleGenerativeAI(
        api_key=os.getenv("gemini_api_key_2"), # type: ignore
        model="gemini-1.5-pro",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

# Define the template for the test case generation prompt
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

# Define the model for request data
class TestCaseRequest(BaseModel):
    user_story: str
    jira_id: str
    acceptance_criteria: Optional[str] = None

# Define the model for response data
class TestCaseResponse(BaseModel):
    jira_id: str
    content: str
    token_count: int

# Simplified HTML for the landing page
landing_page_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Case Generator</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e1e4e8;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .subheading {
            color: #7f8c8d;
            font-size: 16px;
            margin-top: 0;
        }
        .main-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 25px;
            margin-bottom: 30px;
        }
        form {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #34495e;
        }
        textarea, input {
            width: 100%;
            padding: 12px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            transition: border 0.3s ease;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #3498db;
        }
        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
            width: 100%;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .result-container {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
            max-height: 600px;
            overflow-y: auto;
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            color: #7f8c8d;
        }
        .status-bar {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .status-text {
            font-size: 14px;
            font-weight: 500;
        }
        footer {
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 20px;
        }
        .markdown-content h3 {
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .markdown-content h4 {
            color: #3498db;
        }
        .markdown-content h5 {
            color: #16a085;
        }
        .markdown-content ul {
            padding-left: 20px;
        }
        .copy-btn {
            background-color: #e7e9ec;
            color: #333;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 10px;
            float: right;
        }
        .copy-btn:hover {
            background-color: #d5d8dc;
        }
    </style>
</head>
<body>
    <header>
        <h1>Test Case Generator</h1>
        <p class="subheading">Generate comprehensive test scenarios and test cases from user stories</p>
        <div class="status-bar">
            <div class="status-indicator" id="statusIndicator" style="background-color: #e74c3c;"></div>
            <span class="status-text" id="statusText">Checking API status...</span>
        </div>
    </header>

    <div class="main-container">
        <form id="testCaseForm">
            <div>
                <label for="jiraId">JIRA ID*</label>
                <input type="text" id="jiraId" name="jiraId" placeholder="e.g., PROJ-123" required>
            </div>
            <div>
                <label for="userStory">User Story*</label>
                <textarea id="userStory" name="userStory" rows="4" placeholder="As a user, I want to..." required></textarea>
            </div>
            <div>
                <label for="acceptanceCriteria">Acceptance Criteria (optional)</label>
                <textarea id="acceptanceCriteria" name="acceptanceCriteria" rows="4" placeholder="1. First criteria&#10;2. Second criteria"></textarea>
            </div>
            <button type="submit" class="btn" id="generateBtn">Generate Test Cases</button>
        </form>
    </div>

    <div class="result-container" id="result" style="display: none;">
        <button id="copyBtn" class="copy-btn">Copy to Clipboard</button>
        <div id="resultContent" class="markdown-content"></div>
    </div>

    <footer>
        <p>Test Case Generator v1.0</p>
    </footer>

    <script>
        // Check API status
        function checkApiStatus() {
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    const statusIndicator = document.getElementById('statusIndicator');
                    const statusText = document.getElementById('statusText');
                    
                    if (data.status === 'healthy') {
                        statusIndicator.style.backgroundColor = '#2ecc71';
                        statusText.innerText = 'API Online';
                    } else {
                        statusIndicator.style.backgroundColor = '#e74c3c';
                        statusText.innerText = 'API Offline';
                    }
                })
                .catch(error => {
                    document.getElementById('statusIndicator').style.backgroundColor = '#e74c3c';
                    document.getElementById('statusText').innerText = 'API Offline';
                });
        }
        
        // Call immediately and then every 30 seconds
        checkApiStatus();
        setInterval(checkApiStatus, 30000);

        // Handle form submission
        document.getElementById('testCaseForm').addEventListener('submit', function(event) {
            event.preventDefault();
            
            const jiraId = document.getElementById('jiraId').value;
            const userStory = document.getElementById('userStory').value;
            const acceptanceCriteria = document.getElementById('acceptanceCriteria').value;
            const generateBtn = document.getElementById('generateBtn');
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultDiv.style.display = 'block';
            resultContent.innerHTML = '<div class="loading">Generating test cases... This may take a minute or two.</div>';
            generateBtn.disabled = true;
            generateBtn.innerText = 'Generating...';
            
            fetch('/generate-test-cases', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    jira_id: jiraId,
                    user_story: userStory,
                    acceptance_criteria: acceptanceCriteria
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Use marked.js to render markdown
                resultContent.innerHTML = marked.parse(data.content);
                generateBtn.disabled = false;
                generateBtn.innerText = 'Generate Test Cases';
            })
            .catch(error => {
                resultContent.innerHTML = `<div style="color: #e74c3c; font-weight: bold;">Error: ${error.message}</div>`;
                generateBtn.disabled = false;
                generateBtn.innerText = 'Generate Test Cases';
            });
        });
        
        // Copy to clipboard functionality
        document.getElementById('copyBtn').addEventListener('click', function() {
            const resultContent = document.getElementById('resultContent');
            const range = document.createRange();
            range.selectNode(resultContent);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();
            
            const originalText = this.innerText;
            this.innerText = 'Copied!';
            setTimeout(() => {
                this.innerText = originalText;
            }, 2000);
        });
    </script>
</body>
</html>"""

# Define the endpoint for the landing page
@app.get("/", response_class=HTMLResponse)
async def get_landing_page():
    return landing_page_html

# Define the endpoint for generating test cases
@app.post("/generate-test-cases", response_model=TestCaseResponse)
async def generate_test_cases(request: TestCaseRequest = Body(...)):
    try:
        # Get the LLM
        llm = get_llm()
        
        # Format the prompt
        formatted_prompt = test_case_prompt.format(
            user_story=request.user_story,
            jira_id=request.jira_id,
            acceptance_criteria=request.acceptance_criteria or ""
        )
        
        # Invoke the LLM to generate test cases
        response = llm.invoke(formatted_prompt)
        content = response.content
        
        # Calculate token count (approximation using word count)
        token_count = len(content.split()) # type: ignore
        
        # Return the response
        return TestCaseResponse(
            jira_id=request.jira_id,
            content=content, # type: ignore
            token_count=token_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test cases: {str(e)}")

# Define a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)