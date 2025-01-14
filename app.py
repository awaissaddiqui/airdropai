from flask import Flask, render_template, jsonify
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from phi.agent import Agent
from air_drops_agent import (
    airdrop_discovery_agent,
    multi_ai_agent,
)
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_agents", methods=["GET"])
def run_single_agent():
    """
    Runs the single agent (Airdrop Discovery Agent), formats the results,
    and sends them to the frontend.
    """
    try:
        # Step 1: Run the single agent
        raw_results = airdrop_discovery_agent.run(
            "Find ongoing cryptocurrency airdrops and provide initial evaluations."
        )

        # Step 2: Format results with LangChain
        formatted_results = format_results_with_langchain(
            raw_results, agent_type="Single Agent"
        )

        # Step 3: Return results to the frontend
        return jsonify({"results": formatted_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run_agents_multi", methods=["GET"])
def run_multi_agent():
    """
    Runs the multi-agent system, formats the results, and sends them to the frontend.
    """
    try:
        # Step 1: Run the multi-agent system
        raw_results = multi_ai_agent.run(
            "Find ongoing airdrops, evaluate their legitimacy, and provide secure claiming instructions."
        )

        # Step 2: Format results with LangChain
        formatted_results = format_results_with_langchain(
            raw_results, agent_type="Multi-Agent"
        )

        # Step 3: Return results to the frontend
        return jsonify({"results": formatted_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def format_results_with_langchain(raw_results, agent_type):
    """
    Formats the raw results using LangChain's ChatOpenAI integration.
    """
    try:
        # Adjust raw_results to extract the markdown content
        if isinstance(raw_results, dict) and "content" in raw_results:
            raw_data_content = raw_results["content"]
        else:
            raw_data_content = raw_results

        # Create proper LangChain message objects
        messages = [
            SystemMessage(content=f"""You are an AI assistant processing cryptocurrency airdrop data from a {agent_type} system.
            Format the data into a structured markdown report with clear sections for each airdrop and important notes.
            Follow this exact format:

            # Ongoing Cryptocurrency Airdrops Report

            ## [Airdrop Name]
            **Legitimacy**: [Detailed evaluation of the airdrop's legitimacy]

            **Claim Instructions**:
            1. [Step 1 with clear instructions]
            2. [Step 2 with clear instructions]
            3. [Additional steps as needed]

            ---

            [Repeat the above section for each airdrop]

            ---

            ## Important Notes
            - [Note about security]
            - [Note about eligibility]
            - [Note about verification]
            - [Additional important notes]

            Each section must be separated by exactly three dashes (---) for proper parsing.
            """),
            HumanMessage(content=f"Here is the raw data about cryptocurrency airdrops:\n\n{raw_data_content}")
        ]

        # Generate the response using the ChatOpenAI model
        response = llm.invoke(messages)
        
        # Return the content of the response
        return response.content
    except Exception as e:
        return f"Error in LangChain formatting: {e}"

