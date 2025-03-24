from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Load Ollama Model
model = OllamaLLM(model="llama3.2")

# Define Prompt Template
prompt = PromptTemplate.from_template("""
    You are a QA Engineer expert in generating test case for any provided user story. Your task as an experienced QA Engineer is to provide the Test Scenarios for the given user stories in an easy to understand but descriptive Gherkin Syntax. The user stories will vary according to the user input, so accordingly change the test scenarios also. Provide no additional helping text and just provide me with the Gherkin documentation. Make sure to add every positive and negative scenarios for detailed testing. Provide the output in markdown format with Gherkin highlighting as well.
    
    Here is the user Story.
    {user_story}
""")

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    try:
        # Get the JSON data from request
        data = request.json
        user_story = data.get("user_story", "")

        if not user_story:
            return jsonify({"error": "User story is required"}), 400

        # Generate test cases
        response = model.invoke(prompt.format(user_story=user_story))

        return jsonify({"gherkin_test_cases": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)