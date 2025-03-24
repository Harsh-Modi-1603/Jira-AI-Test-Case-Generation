from dataclasses import dataclass
import json
import logging
from typing import List, Dict, Optional
from jira import JIRA
from jira.exceptions import JIRAError
from langchain_ollama import OllamaLLM
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class JiraStory:
    key: str
    summary: str
    description: str
    acceptance_criteria: str

@dataclass
class TestCase:
    title: str
    preconditions: List[str]
    steps: List[str]
    expected_results: List[str]

class JiraConnector:
    def __init__(self, server: str, email: str, token: str):
        try:
            self.jira = JIRA(server=server, basic_auth=(email, token))
            logger.info(f"Successfully connected to JIRA server: {server}")
        except Exception as e:
            logger.error(f"Failed to connect to JIRA: {str(e)}")
            raise
    
    def get_story(self, key: str) -> Optional[JiraStory]:
        try:
            issue = self.jira.issue(key)
            logger.info(f"Successfully retrieved JIRA story: {key}")
            
            # Get custom field ID for acceptance criteria
            custom_fields = self.jira.fields()
            acceptance_criteria_field = next(
                (field['id'] for field in custom_fields 
                 if 'acceptance criteria' in field['name'].lower()),
                'customfield_10000'  # fallback to default
            )
            
            return JiraStory(
                key=issue.key,
                summary=issue.fields.summary,
                description=issue.fields.description or "",
                acceptance_criteria=getattr(issue.fields, acceptance_criteria_field, "") or ""
            )
        except JIRAError as e:
            logger.error(f"Failed to retrieve JIRA story {key}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while retrieving story {key}: {str(e)}")
            return None

class TestCaseGenerator:
    def __init__(self, model_name: str = "llama3.2"):
        try:
            self.model = model_name
            self.llm = OllamaLLM(model=self.model)
            logger.info(f"Successfully initialized LLM with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text input"""
        return text.strip().replace('\r\n', '\n').replace('\r', '\n')

    def extract_requirements(self, story: JiraStory) -> List[str]:
        try:
            context = f"""
            Please analyze this user story and extract key testable requirements:
            
            Story: {story.summary}
            Description: {self._clean_text(story.description)}
            Acceptance Criteria: {self._clean_text(story.acceptance_criteria)}
            
            Format your response as a list of clear, testable requirements.
            Each requirement should be atomic and independently verifiable.
            """
            
            response = self.llm.invoke(context)
            requirements = [req.strip() for req in response.split('\n') if req.strip()]
            logger.info(f"Extracted {len(requirements)} requirements from story {story.key}")
            return requirements
        except Exception as e:
            logger.error(f"Failed to extract requirements: {str(e)}")
            return []

    def generate_test_cases(self, requirements: List[str]) -> List[TestCase]:
        test_cases = []
        for req in requirements:
            try:
                prompt = f"""
                Generate a test case in Gherkin format for this requirement:
                {req}
                
                Use this format:
                Scenario: [Clear title describing the test case]
                Given [precondition]
                When [action]
                Then [expected result]
                
                Make sure to include all necessary steps and validations.
                """
                
                response = self.llm.invoke(prompt)
                test_case = self._parse_gherkin(response)
                if test_case:
                    test_cases.append(test_case)
                    logger.info(f"Generated test case: {test_case.title}")
            except Exception as e:
                logger.error(f"Failed to generate test case for requirement: {req}\nError: {str(e)}")
        
        return test_cases

    def _parse_gherkin(self, gherkin: str) -> Optional[TestCase]:
        try:
            lines = [line.strip() for line in gherkin.split('\n') if line.strip()]
            
            # Find scenario title
            title_line = next((line for line in lines if line.lower().startswith('scenario:')), None)
            if not title_line:
                logger.warning("No scenario title found in Gherkin output")
                return None
            
            title = title_line.replace('Scenario:', '').strip()
            
            # Extract steps
            given = [line.replace('Given', '').strip() for line in lines if line.strip().lower().startswith('given')]
            when = [line.replace('When', '').strip() for line in lines if line.strip().lower().startswith('when')]
            then = [line.replace('Then', '').strip() for line in lines if line.strip().lower().startswith('then')]
            
            return TestCase(
                title=title,
                preconditions=given,
                steps=when,
                expected_results=then
            )
        except Exception as e:
            logger.error(f"Failed to parse Gherkin: {str(e)}")
            return None

class TestCaseExporter:
    @staticmethod
    def to_gherkin(test_cases: List[TestCase], output_file: Optional[Path] = None) -> str:
        output = []
        for tc in test_cases:
            output.append(f"Feature: {tc.title}")
            output.append(f"Scenario: {tc.title}\n")
            for pre in tc.preconditions:
                output.append(f"  Given {pre}")
            for step in tc.steps:
                output.append(f"  When {step}")
            for result in tc.expected_results:
                output.append(f"  Then {result}")
            output.append("\n")
        
        result = "\n".join(output)
        
        if output_file:
            try:
                output_file.write_text(result)
                logger.info(f"Successfully wrote Gherkin output to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write Gherkin output to file: {str(e)}")
        
        return result
    
    @staticmethod
    def to_json(test_cases: List[TestCase], output_file: Optional[Path] = None) -> str:
        try:
            json_output = json.dumps([vars(tc) for tc in test_cases], indent=2)
            
            if output_file:
                try:
                    output_file.write_text(json_output)
                    logger.info(f"Successfully wrote JSON output to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to write JSON output to file: {str(e)}")
            
            return json_output
        except Exception as e:
            logger.error(f"Failed to convert test cases to JSON: {str(e)}")
            return ""

def main():
    try:
        # Initialize components
        jira = JiraConnector(
            server="https://inappad.atlassian.net",
            email="harsh.m@iqm.com",
            token=os.getenv("JIRA_TOKEN") # type: ignore
        )
        
        generator = TestCaseGenerator()
        exporter = TestCaseExporter()
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Process a story
        story = jira.get_story("RD-294")
        if not story:
            logger.error("Failed to retrieve story")
            return
        
        requirements = generator.extract_requirements(story)
        if not requirements:
            logger.error("No requirements extracted from story")
            return
        
        test_cases = generator.generate_test_cases(requirements)
        if not test_cases:
            logger.error("No test cases generated")
            return
        
        # Export results
        gherkin_output = exporter.to_gherkin(
            test_cases, 
            output_file=output_dir / f"{story.key}_test_cases.feature"
        )
        json_output = exporter.to_json(
            test_cases,
            output_file=output_dir / f"{story.key}_test_cases.json"
        )
        
        print("\nGenerated Test Cases (Gherkin format):")
        print(gherkin_output)
        
        print("\nGenerated Test Cases (JSON format):")
        print(json_output)
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()