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

# HTML for the landing page
landing_page_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Case Generator API</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        h1 {
            color: #2c3e50;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
        }
        .main-content {
            flex: 2;
            min-width: 300px;
        }
        .sidebar {
            flex: 1;
            min-width: 300px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        pre {
            background-color: #f6f8fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            font-family: 'Courier New', Courier, monospace;
        }
        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        form {
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            background-color: #2ecc71;
            color: white;
        }
        .api-endpoint {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 5px solid #3498db;
        }
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            max-height: 600px;
            overflow-y: auto;
        }
        #resultContent {
            white-space: pre-wrap;
            font-family: 'Courier New', Courier, monospace;
            font-size: 14px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <header>
        <h1>Test Case Generator API</h1>
        <p>Generate comprehensive test scenarios and test cases from JIRA user stories</p>
        <span class="status">API Status: <span id="apiStatus">Checking...</span></span>
    </header>

    <div class="container">
        <div class="main-content">
            <h2>About This Service</h2>
            <p>
                This API service uses advanced AI to generate detailed test scenarios and test cases based on your JIRA user stories.
                The generated test cases follow a structured format with comprehensive coverage of positive, negative, and edge cases.
            </p>

            <h2>API Endpoints</h2>
            
            <div class="api-endpoint">
                <h3>Generate Test Cases</h3>
                <p><strong>POST</strong> /generate-test-cases</p>
                <p>Generates test cases based on the provided user story and saves the output to a file.</p>
                <h4>Request Body:</h4>
                <pre><code>{
  "user_story": "As a user, I want to...",
  "jira_id": "PROJ-123",
  "acceptance_criteria": "1. Criteria one\\n2. Criteria two"
}</code></pre>
                <h4>Response:</h4>
                <pre><code>{
  "jira_id": "PROJ-123",
  "content": "Generated test cases in markdown format...",
  "token_count": 3500
}</code></pre>
            </div>

            <div class="api-endpoint">
                <h3>Health Check</h3>
                <p><strong>GET</strong> /health</p>
                <p>Returns the health status of the API.</p>
                <h4>Response:</h4>
                <pre><code>{
  "status": "healthy"
}</code></pre>
            </div>

            <h2>Documentation</h2>
            <p>
                For complete API documentation, visit the <a href="/docs">Swagger UI</a> or <a href="/redoc">ReDoc</a> documentation.
            </p>
        </div>

        <div class="sidebar">
            <h2>Try It Out</h2>
            <form id="testCaseForm">
                <div>
                    <label for="jiraId">JIRA ID:</label>
                    <input type="text" id="jiraId" name="jiraId" placeholder="e.g., PROJ-123" required>
                </div>
                <div>
                    <label for="userStory">User Story:</label>
                    <textarea id="userStory" name="userStory" rows="4" placeholder="As a user, I want to..." required></textarea>
                </div>
                <div>
                    <label for="acceptanceCriteria">Acceptance Criteria (optional):</label>
                    <textarea id="acceptanceCriteria" name="acceptanceCriteria" rows="4" placeholder="1. First criteria&#10;2. Second criteria"></textarea>
                </div>
                <button type="submit" class="btn">Generate Test Cases</button>
            </form>
            <div id="result" style="display: none;">
                <h3>Result</h3>
                <div id="resultContent"></div>
            </div>
        </div>
    </div>

    <h2>Example Output</h2>
    <pre><code>### **User Story**  
**Story Title:** Update email notification to use organization timezone  
**Description:** As a user, In the received Audience File Uploaded notification email the date should belong to the Timezone of the Organization not UTC.  
**JIRA Issue ID:** AUD-1374  

### **Acceptance Criteria**  
1. The mentioned date in email must be from the Organization profile timezone 
2. The organization profile must follow the ET timezone by default if not set
3. The text must contain Date and Timezone both.

---
### **Test Scenarios & Test Cases**  

#### **Test Scenario ID: TS_01**  
**Test Scenario:** Validate whether the email notification displays dates in the organization's timezone

##### **Test Case ID: TC_01**  
- **Test Case:** Validate whether the email notification displays dates in the organization's configured timezone
- **Preconditions:** 
  1. Organization has a configured timezone in their profile
  2. User has permission to receive email notifications
  3. An audience file has been uploaded
- **Test Data:** 
  The test data is just for guidance and the actual test data is to be determined by the user.
  - Organization timezone: PST (UTC-8)
  - Uploaded file: audience_data.csv
  - Upload time: 2023-05-15 10:30:00 UTC
...
</code></pre>

    <footer>
        <p>Test Case Generator API v1.0 | © 2025</p>
    </footer>

    <script>
        // Check API status
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                document.getElementById('apiStatus').innerText = data.status;
                document.getElementById('apiStatus').style.color = 'white';
            })
            .catch(error => {
                document.getElementById('apiStatus').innerText = 'Offline';
                document.getElementById('apiStatus').style.color = 'red';
            });

        // Handle form submission
        document.getElementById('testCaseForm').addEventListener('submit', function(event) {
            event.preventDefault();
            
            const jiraId = document.getElementById('jiraId').value;
            const userStory = document.getElementById('userStory').value;
            const acceptanceCriteria = document.getElementById('acceptanceCriteria').value;
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultDiv.style.display = 'block';
            resultContent.innerText = 'Generating test cases... This may take a minute or two.';
            
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
            })
            .catch(error => {
                resultContent.innerText = `Error: ${error.message}`;
            });
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