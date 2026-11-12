
# AI Usage

The assignment encourages AI assistance and asks candidates to push their AI coding configuration. This file documents how AI was used throughout this project and how its output was verified.

## Tools and configuration

- **Assistant:** Claude (Anthropic), used through the Claude web chat interface.
- **Configuration:** None to push. No IDE agent (Copilot, Cursor), MCP servers, custom skills or rules files were used for coding. All code was written in a regular editor and run locally.
- **In the application itself:** The agent calls an OpenAI-compatible model through the LiteLLM proxy provided by Syntax. This is the product's runtime, not a coding tool, and is documented in the README.

## How AI was used

I drove the engineering process. I chose the architecture, defined the requirements and made the key design decisions. Before implementation, I discussed the proposed architecture and alternatives with Claude, challenged suggestions, and refined the approach until we agreed on a solution.

Once a decision was made, I asked Claude to implement the corresponding part of the code. We worked incrementally:

1. **Architecture first:** I defined what each component should do, how components should interact, and why a particular approach was appropriate. I discussed alternatives and trade-offs with Claude before settling on decisions.
2. **Implementation second:** I asked Claude to implement one component or change at a time, with explanations of the code and the reasoning behind it.
3. **Testing together:** I ran the commands and tests locally, examined the actual output, and shared the results with Claude. We discussed failures, investigated their causes and agreed on fixes before proceeding.
4. **Controlled progression:** The next step only started after the current one worked as expected and was committed.

Claude proposed designs, wrote code and helped debug, but I remained involved in the technical discussions, decisions, implementation reviews and validation. The Git history, with commits for individual steps and subsequent fixes, reflects this incremental workflow.

## How the output was verified

I reviewed and tested the work rather than accepting AI-generated output on trust:

- **Every component was tested before moving on.** The repository has 134 automated tests, and every step ended with the tests passing.
- **Decisions were based on measurements, not assumptions.** For example, the planner was only built after a failing retrieval case had been reproduced and the proposed fix proven with existing code. A latency optimisation was measured, showed no gain, and was reverted.
- **Answers were checked against the corpus itself.** During a 29-question test series, claims were verified line by line in the source files (using `Select-String`) before a result was accepted.
- **Assumptions were challenged.** Several suspected bugs turned out to be display artifacts or provider latency spikes, and were proven so before any code was changed. Conversely, a suspected retrieval problem turned out to be an answer-generation problem, and the fix was made there.

## Decisions I made or changed

The assistant's suggestions were not always followed as proposed. Examples include:

- Accepting both `OPENAI_BASE_URL` and the misspelled `OPENAI_BASE_UR` from the provided `.env` (my proposal).
- Building only what each step needed instead of preparing configuration "just in case", including growing `.gitignore` as files appeared.
- Requiring the drug filter to be strict while section filters stay a preference.
- Requiring that a citation to a missing source and a citation that does not support its claim be treated as different problems.
- Requiring that every revision be followed by a new review, with a bounded number of revisions.
- Investigating retrieval ranking before changing the answer generator on the albuterol case, and checking whole source files before concluding that information was missing.
- Reverting a prompt change that shortened answers once measurements showed it did not reduce latency.

The goal was not to avoid AI-generated code, but to use AI as an implementation and reasoning partner while retaining ownership of the architecture, technical decisions and validation.
