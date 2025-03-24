from jira import JIRA
from dotenv import load_dotenv
import os
import json


load_dotenv()

email = os.getenv("JIRA_EMAIL")
token = os.getenv("JIRA_TOKEN")

jira = JIRA(server="https://inappad.atlassian.net/", basic_auth=(email, token))
issue = jira.issue("RD-294")

# with open("jira-output.json","w") as file:
#     json.dump(issue.raw, file)


print(issue.fields.summary)