from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """
    What is the capital of {country},
    What is the GDP of {city},
    """
)

delhi = template.format(country="Delhi", city="Lahore")
gujarat = template.format(country = "Gujarat",city="Bhavnagar")

print(delhi, gujarat)