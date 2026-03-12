import os
from crewai import Crew, Process, Task
from agents import researcher, analyst, manager

# Define tasks directly for simplicity
research_task = Task(
    description='Search for news regarding {ticker} from the last 7 days.',
    expected_output='A summary of the top news stories.',
    agent=researcher
)

technical_task = Task(
    description='Run technical analysis for {ticker}.',
    expected_output='A report containing current price and SMA.',
    agent=analyst
)

# Assemble the Crew
financial_crew = Crew(
    agents=[researcher, analyst, manager],
    tasks=[research_task, technical_task],
    process=Process.sequential
)

if __name__ == "__main__":
    result = financial_crew.kickoff(inputs={'ticker': 'RELIANCE.NS'})
    print(result)
