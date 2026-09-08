"""Simple connectivity check for a Pydantic AI agent.

This script connects to local Ollama or the shared HEIA vLLM endpoint. It is
useful as a smoke test before running more complex RAG examples.

The script demonstrates the following steps:
1. Define a Pydantic AI agent with basic instructions and a model.
2. Run a simple test prompt to check that the agent can generate a response.
3. Optionally, expose the agent as a local web app.
"""

from time import perf_counter

import uvicorn
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

#################################################################
# STEP 1 - Define the Pydantic AI agent
#################################################################


# The default provider is local Ollama. Set LLM_PROVIDER=vllm to use the shared
# HEIA Kubernetes deployment instead.
agent = Agent(
    model=get_llm_model(),
    instructions="You are a helpful assistant.",
    model_settings=ModelSettings(thinking="minimal"),
)

#################################################################
# STEP 2 - Run a simple smoke test to check that the agent can generate a response
#################################################################


# Define a simple smoke test function to check that the agent can generate a response.
def smoke_test() -> None:
    """Run a simple test prompt to check that the agent works.

    This function sends a basic question to the model and prints the answer.
    It also measures the response time and logs the approximate number of
    output tokens generated per second.

    Returns:
        None.
    """
    print_step("Pydantic AI Agent - Entry Point Demo")

    # Start measuring execution time.
    start = perf_counter()

    # Send a simple prompt to the agent.
    result = agent.run_sync("What is the capital of Italy?")

    # Compute the total generation time.
    duration = perf_counter() - start

    # Print an approximate speed value.
    console.print(
        f"Token per second: {result.usage().output_tokens / duration:.2f} tokens/s",
        style=INFO_STYLE,
    )

    # Print the model response.
    print_step("Agent Output")
    print_result(result.output)


# Run the smoke test when the script is executed.
if __name__ == "__main__":
    smoke_test()

    #################################################################
    # STEP 3 - Expose the agent as a local web app
    #################################################################

    app = agent.to_web()
    console.print(
        "Starting Simple Pydantic AI Agent on http://127.0.0.1:8000",
        style=INFO_STYLE,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
