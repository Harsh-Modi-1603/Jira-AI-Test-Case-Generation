from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from collections import defaultdict
from typing import Optional
import os
from dotenv import load_dotenv
from scripts.llm import test_case_prompt, llm_model
from jira import JIRA
from jira.exceptions import JIRAError
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import hashlib


load_dotenv()

app = FastAPI(
    title="Test Case Generator API",
    description="API for generating test cases based on user stories",
)
# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def health_check():
    return "JIRA API Server running!"


class IssueFetchRequest(BaseModel):
    domain: str
    email: str
    jira_id: str  # This will be the Epic ID
    jira_token: str


class StoryItem(BaseModel):
    key: str
    summary: str
    description: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    epic_link: Optional[str] = None
    tags: list[str] = []


class IssueFetchResponse(BaseModel):
    epic_key: str
    epic_summary: str
    epic_description: Optional[str] = None
    stories: list[StoryItem] = []


class TestCaseRequest(BaseModel):
    user_story: str
    jira_id: str
    acceptance_criteria: Optional[str] = None


cached_dict = defaultdict(dict)


@app.post("/authenticate")
async def authenticate_jira(request: IssueFetchRequest = Body(...)):
    try:
        jira = JIRA(
            server=request.domain, basic_auth=(request.email, request.jira_token)
        )
        # Verify credentials by calling myself()
        user = jira.myself()

        # Store the authenticated JIRA client in cache for later use
        cache_key = f"{request.domain}:{request.email}"
        cached_dict[cache_key] = jira

        return {"status": "authenticated", "username": user.get("displayName")}
    except JIRAError as e:
        if e.status_code == 401:
            raise HTTPException(
                status_code=401, detail="Authentication failed: Invalid credentials"
            )
        raise HTTPException(
            status_code=e.status_code or 500, detail=f"JIRA Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error connecting with JIRA: {str(e)}"
        )


@app.post("/fetch-stories")
async def fetch_epic_stories(request: IssueFetchRequest = Body(...)):
    try:
        # Get cached JIRA client or create new one if not cached
        cache_key = f"{request.domain}:{request.email}"
        jira = cached_dict.get(cache_key)

        if not jira:
            jira = JIRA(
                server=request.domain, basic_auth=(request.email, request.jira_token)
            )
            cached_dict[cache_key] = jira

        epic = jira.issue(request.jira_id)

        jql_query = (
            f'"Epic Link" = {request.jira_id} AND issuetype = Story ORDER BY key ASC'
        )
        stories = jira.search_issues(jql_query, maxResults=100)

        story_items = []
        for story in stories:
            # Extract labels/tags
            tags = story.fields.labels if hasattr(story.fields, "labels") else []

            # Extract assignee name
            assignee = (
                story.fields.assignee.displayName if story.fields.assignee else None
            )

            # Extract priority, status, due date
            priority = (
                story.fields.priority.name
                if hasattr(story.fields.priority, "name")
                else None
            )
            status = story.fields.status.name if hasattr(story.fields, "name") else None
            due_date = (
                story.fields.duedate if hasattr(story.fields, "duedate") else None
            )

            story_items.append(
                StoryItem(
                    key=story.key,
                    summary=story.fields.summary,
                    description=story.fields.description,
                    priority=priority,
                    status=status,
                    assignee=assignee,
                    due_date=due_date,
                    epic_link=request.jira_id,
                    tags=tags,
                )
            )

        return IssueFetchResponse(
            epic_key=epic.key,
            epic_summary=epic.fields.summary,
            epic_description=epic.fields.description,
            stories=story_items,
        )
    except JIRAError as e:
        if e.status_code == 401:
            raise HTTPException(
                status_code=401, detail="Authentication failed: Invalid credentials"
            )
        raise HTTPException(
            status_code=e.status_code or 500, detail=f"JIRA Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error connecting with JIRA: {str(e)}"
        )


def create_cache_key(user_story, jira_id, acceptance_criteria):
    combined = f"{user_story}|{jira_id}|{acceptance_criteria}"
    return hashlib.md5(combined.encode()).hexdigest()


test_case_cache = {}


@app.post("/generate-test-cases")
async def generate_test_cases(request: TestCaseRequest = Body(...)):
    try:
        # Create a cache key based on input parameters
        cache_key = create_cache_key(
            request.user_story, request.jira_id, request.acceptance_criteria or ""
        )

        # Check if we have a cached response
        if cache_key in test_case_cache:
            return test_case_cache[cache_key]

        formatted_prompt = test_case_prompt.format(
            user_story=request.user_story,
            jira_id=request.jira_id,
            acceptance_criteria=request.acceptance_criteria or "",
        )

        response = llm_model.invoke(formatted_prompt)
        content = response.content
        token_count = len(content.split())

        # Cache the response
        result = {"content": content, "token_count": token_count}
        test_case_cache[cache_key] = result

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating test cases: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
