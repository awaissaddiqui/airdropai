from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
os.environ["OPENAI_API_KEY"]="sk-svcacct-tdotXkiAMpgrv6IrVbZ5AhGaF9zUKID2jRi2M0QVtpLkQNtjXMYCrTGIWupy_qOeLT3BlbkFJAH8qsq480oaZSwYnmDyRRdXs0mQWiQGP7PngN9G3Q2DH0gIq3KbvkGQvaylEkhEAA"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



# Debug: Ensure keys are loaded
if not NEWS_API_KEY:
    print("API keys are missing. Check your .env file.")
else:
    print("API keys loaded successfully.")

# Define Agents
airdrop_discovery_agent = Agent(
    name="Airdrop Discovery Agent",
    role="Discover ongoing cryptocurrency airdrops",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGo(), Newspaper4k()],
    instructions=[
        "Search for ongoing cryptocurrency airdrops using DuckDuckGo and Newspaper4k.",
        "Focus on sources discussing Ethereum, Polygon, Solana, and other popular blockchain ecosystems.",
        "Filter out irrelevant or suspicious results and prioritize reputable projects.",
        "Present the results in a table with columns for Name, Source, Details, and Initial Evaluation.",
    ],
    show_tool_calls=True,
    markdown=True,
)

legitimacy_agent = Agent(
    name="Legitimacy Assessment Agent",
    role="Evaluate the legitimacy of cryptocurrency airdrops",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGo(), Newspaper4k()],
    instructions=[
        "Analyze the legitimacy of cryptocurrency airdrops identified by the Discovery Agent.",
        "Provide a risk score (1-10) and a detailed evaluation for each airdrop.",
        "Present findings in a table with columns for Name, Risk Score, Details, and Final Verdict.",
    ],
    show_tool_calls=True,
    markdown=True,
)

guidance_agent = Agent(
    name="Step-by-Step Guidance Agent",
    role="Provide instructions for claiming cryptocurrency airdrops",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[],
    instructions=[
        "Provide step-by-step instructions for claiming cryptocurrency airdrops.",
        "Include wallet setup guidance, transaction fee details, and security precautions.",
        "Present instructions in a markdown list format.",
    ],
    show_tool_calls=True,
    markdown=True,
)

security_agent = Agent(
    name="Security and Compliance Agent",
    role="Ensure compliance and security for cryptocurrency airdrops",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[],
    instructions=[
        "Provide guidelines for KYC/AML compliance and wallet integration security.",
        "Highlight security best practices for users, such as avoiding sharing private keys or seed phrases.",
        "Monitor evolving regulations and adapt recommendations to ensure compliance.",
        "Present recommendations in a bullet point list format.",
    ],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[
        airdrop_discovery_agent,
        legitimacy_agent,
        guidance_agent,
        security_agent,
    ],
    instructions=[
        "Collaborate to identify and evaluate cryptocurrency airdrops.",
        "Provide a risk score, detailed claiming instructions, and security tips.",
        "Always include sources and use tables to organize information.",
    ],
    show_tool_calls=True,
    markdown=True,
)

def run_agents():
    try:
        print("Running Multi-Agent System...")
        results = multi_ai_agent.run(
            "Find ongoing airdrops, evaluate their legitimacy, and provide secure claiming instructions."
        )
        print("Execution Successful!")
        return results
    except Exception as e:
        print(f"Error during execution: {e}")
        if "grpc_wait_for_shutdown_with_timeout" in str(e):
            print("This error suggests a gRPC timeout. Check network connectivity or increase timeout settings.")
        return {"error": str(e), "suggestion": "Check gRPC server or network connection."}
