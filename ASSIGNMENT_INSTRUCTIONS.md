#### **To get started, accept the assignment on Github Classroom and get the starter code**


## 1) Objective

You will design, implement, and evaluate a multi-agent system for **deep research** on an HCI-relevant topic. Your system should:

- Orchestrate agents using **AutoGen** or **LangGraph** (your choice, or both).  
  (We provide a free self-hosted GPT-OSS-20B model endpoint; more information will be given during the Nov. 21 class.)
- Provide a simple **user interface** (CLI or web) for interactive querying.
- Include **safety guardrails** to handle unsafe/inappropriate requests (e.g., Guardrails, NeMo Guardrails, or equivalent policy filtering).
- Evaluate generated outputs using **LLM-as-a-Judge** scoring and report findings.

The deliverables are a **technical report** (3–4 page single-column and single-space write-up) and a **GitHub Classroom repo** containing your code and documentation.

To get started with the assignment:

- Accept the assignment and get the starter code.
- Read through the code to understand the framework, and check the **implementation-guide** section to understand where and how to implement the changes needed.
- In general, we have specified `TODO: YOUR CODE HERE` around the code base. As long as you implement correct changes in place, the system should be good to go by the end.

### Implementation Guide

If you need some pointers for the implementation, below are what to look out for.
You do **not** need to rewrite everything from scratch. In most cases, you should extend the provided structure and fill in the marked implementation areas.

- **Agent roles and orchestration**
  - Start in `src/agents/autogen_agents.py` if you are using the provided AutoGen scaffold.
  - The TODOs here should define what each agent is responsible for, what prompt/instructions it uses, and which tools it can access.
  - Then connect those agents in `src/autogen_orchestrator.py`.
  - The TODOs in the orchestrator should define the workflow order, how messages move between agents, when to stop, and how to package the final result.

- **Tool use and evidence collection**
  - Implement search and evidence helpers in `src/tools/`.
  - In `src/tools/web_search.py`, the TODOs should connect to a real web search provider and return structured, readable results.
  - In `src/tools/paper_search.py`, the TODOs should connect to an academic search source such as Semantic Scholar and return paper metadata.
  - In `src/tools/citation_tool.py`, the TODOs should help format or track sources so students can show evidence clearly in outputs.
  - If you add more tools, keep them modular and make sure the researcher/orchestrator is what calls them.

- **Safety and guardrails**
  - Implement input/output safety checks in `src/guardrails/`.
  - In `src/guardrails/input_guardrail.py`, TODOs should detect unsafe or malicious user input, such as harmful requests, prompt injection attempts, or off-topic queries.
  - In `src/guardrails/output_guardrail.py`, TODOs should inspect model outputs for unsafe content, misinformation risks, or PII, and decide whether to redact or refuse.
  - In `src/guardrails/safety_manager.py`, TODOs should coordinate the overall policy, connect input/output checks, and log safety events in a format the UI can display.
  - Your orchestrator or main query-processing path should call these guardrails before and/or after model generation.

- **User interface**
  - Put interactive entry points in `src/ui/`.
  - In `src/ui/cli.py`, TODOs should implement a usable terminal workflow: input loop, command handling, formatted outputs, traces, citations, and safety messages.
  - In `src/ui/streamlit_app.py`, TODOs should implement the web interface for entering queries, displaying agent traces, showing citations/sources, and surfacing safety refusals or sanitization events.
  - This layer should remain focused on presentation and interaction rather than core research logic.

- **Evaluation and LLM-as-a-Judge**
  - Implement judging and batch evaluation in `src/evaluation/`.
  - In `src/evaluation/judge.py`, TODOs should build the judge prompt(s), call the judge model, parse scores, and return criterion-level feedback.
  - In `src/evaluation/evaluator.py`, TODOs should load a dataset, run the system on multiple queries, collect scores, and generate an aggregate report.
  - Your test queries/datasets should live under `data/`, and this is where you should add more than 5 diverse evaluation prompts.

- **Configuration and reproducibility**
  - Keep tunable settings such as models, prompts, tool providers, safety policies, and evaluation criteria in `config.yaml`.
  - Keep environment-variable setup in `.env` / `.env.example`.
  - In `main.py`, TODOs should provide one or more clean entry points for running the CLI, web UI, and evaluation pipeline.
  - Students should avoid scattering hard-coded settings across files when they belong in config.

- **Documentation and report support**
  - Use `README.md` for setup, run instructions, screenshots, and reproducibility notes.
  - Store exported examples, logs, evaluation outputs, or representative artifacts in clearly named folders such as `outputs/` or `docs/`.
  - The TODOs here are less about code and more about making sure a grader can actually run the system and inspect at least one end-to-end example.
  - Your technical report should explain how the implementation in these files satisfies the assignment requirements below.

## 2) System Requirements

### A. Agents & Orchestration (AutoGen or LangGraph)

- Minimum **3 agents** with clear roles (e.g., *Planner*, *Retriever/Researcher*, *Critic/Verifier*, optional *Safety/Sanitizer*, *Writer*).
- Recommended workflow: task planning → evidence gathering → synthesis → critique/revision → final answer.
- Include **tool use** (e.g., web search/S2 API, citation extraction, PDF parsing) where appropriate. You may mock tools if needed, but explain trade-offs.
  - Some helpful web search services:
    - [https://www.tavily.com/](https://www.tavily.com/) (You can get a student free quota)
    - [https://brave.com/search/api/](https://brave.com/search/api/)
    - For paper search: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)

### B. Safety & Guardrails

- Integrate at least one safety framework, e.g., [**Guardrails**](https://github.com/guardrails-ai/guardrails), **NeMo Guardrails**, or a custom policy filter.
- System must **detect and handle** unsafe inputs and potential unsafe outputs (refuse, route to safe alternative, or redact).
- Detail your **guardrail policy** or documented list of prohibited categories and response strategies in your write-up.
- Log safety events (what was blocked/redacted and why); this should be communicated in both logs and user interfaces (see below).

### C. User Interface

- Provide a minimal but usable interface:
  - **CLI** with clear prompts **or** a **web UI** (e.g., Streamlit, Gradio, minimal React/Flask/FastAPI).
  - Display **agents' output traces**
    - Show **citations/evidence** collected by the system.
  - Indicate when a response was **refused or sanitized** due to safety policies.

### D. Evaluation: LLM-as-a-Judge

- Define **task prompts** and **ground-truth/expectation criteria** for your topic.
  - Some topics you can consider:
    - *Literature reconnaissance on an HCI concept (e.g., explainable AI for novices, ethical AI in education, AR usability).*
    - *Comparative review of design patterns from research papers, blogs, and docs.*
    - *Synthesis of best practices for UI components from mixed sources (docs, repos, articles).*
    - *Trend analysis and critique of a recent HCI subarea (e.g., agentic UX, AI-driven prototyping).*

- Use at least **2 independent judging prompts** (e.g., different rubrics or perspectives) to score system outputs on. Below are some examples you can use, but you are encouraged to create your own metrics:
  - **Relevance & coverage** of the query
  - **Evidence use & citation quality**
  - **Factual accuracy/consistency**
  - **Safety compliance** (no unsafe content)
  - **Clarity & organization**

- Report a comprehensive evaluation of your multi-agent system in your write-up.

## 3) Report write-up structure

- **Technical Report 3–4 pages, single-column and single-space**
  - **Abstract (~150 words, summarize what you did)**
  - **System Design and implementation** (agents, tools, control flow, models)
  - **Safety Design** (policies, guardrails)
  - **Evaluation Setup and results** (datasets/queries, judge prompts, metrics, error analysis)
  - **Discussion & Limitations** (Summarize your insights and learnings, what are the limitations of this work, and future work)
  - **References** (APA style, **not** counted toward page count)

## 4) Grading Rubric (100 pts)

- **System Architecture & Orchestration (20 pts)**
  - **Agents (10 pts):** Minimum 3 agents with distinct roles; must include planner and researcher; agents must coordinate
  - **Workflow (5 pts):** Clear and well-designed multi-agent workflow
  - **Tools (3 pts):** Web/paper search tools (or other tools) integration
  - **Error Handling (2 pts):** Graceful handling of API failures and invalid inputs

- **User Interface & UX (15 pts)**
  - **Functionality (6 pts):** Working CLI or web interface that accepts queries and displays results
  - **Transparency (6 pts):** Display agent traces, citations/sources, and which agent is active
  - **Safety Communication (3 pts):** Show when content is refused/sanitized

- **Safety & Guardrails (15 pts)**
  - **Implementation (5 pts):** Integrated safety framework with both input and output guardrails
  - **Policies (5 pts):** Documented safety policies (>= 3 categories) integrated into code
  - **Behavior & Logging (5 pts):** System refuses/sanitizes unsafe content and logs events with context

- **Evaluation (LLM-as-a-Judge) (20 pts)**
  - **Implementation (6 pts):** Working judge with >= 2 independent evaluation prompts
  - **Design (6 pts):** >= 3 measurable metrics with clear scoring scales
  - **Analysis (8 pts):** Report evaluation results with interpretation and error analysis. Use more than 5 diverse test queries.

- **Reproducibility & Engineering Quality (10 pts)**  
  Complete README with explanation on how to reproduce the results reported in the write-up.

- **Report Quality and Code Repo (20 pts)**
  - **Structure (8 pts):** 3–4 pages; all required sections; ~150-word abstract; APA references
  - **Content (12 pts):** Clear system design (4 pts), evaluation results (4 pts), discussion of limitations/insights/ethics (4 pts)

- **Bonus (up to +10 pts)**  
  Notable innovation (e.g., toolformer-style augmentation, novel guardrail design, or human eval triangulation).

**Notes — We expect the following from your submission (report + repo):**

- A working web UI interface, e.g. using Streamlit, with the backend multi-agent workflow running. Include:
  - (a) a short demo video or screenshot in the README
  - (b) concise run instructions to reproduce the demo locally
- In your repo, you should also provide a single command or script to run a full end-to-end example with agents communicating with each other (e.g., from query → agents → final synthesis → judge scoring), and document expected outputs/screenshots.
- Specify what query/queries you have tested, and then include:
  - The outputs of different agents and their chat transcripts in your UI; also include an exported sample of at least one full session (JSON) in your repo.
  - The final synthesized answer produced by the system, with inline citations and a separate list of sources. Include at least one exported artifact (e.g., Markdown/HTML) in your repo.
  - **LLM-as-a-Judge results:** Display evaluation results in your UI for at least one run and summarize them in the report. Include raw judge prompts and outputs for at least one representative query in the repo.
  - **Does the guardrail work?:** Ensure your UI indicates when content is refused/sanitized and provide a brief note on which policy category was triggered.