Smolagents Web Browser Agent

This script (`smolagents_scripts/webbrowser.py`) launches a ToolCallingAgent powered browser automation stack (Selenium + Helium) with a suite of tools for navigating, logging in, finding content, downloading files, and building a lightweight per‑site RAG index.

Defaults
- Model: `MODEL_ID` env or `Qwen/Qwen3-4B-Thinking-2507`
- API base: `API_BASE` env or `http://localhost:${PORT:-8000}/v1`
- Headless: false
- Downloads: `downloads/` folder next to `webbrowser.py` (created automatically)
- Max steps: 20
- Verbosity: 2

How to run
- PowerShell (with your venv active and server running):
  - `python smolagents_scripts/webbrowser.py "Go to example.com, summarize the homepage, and download any PDFs."`
  - Optional overrides:
    - `python smolagents_scripts/webbrowser.py --model-id Qwen/Qwen3-4B-Thinking-2507 --api-base http://localhost:8000/v1 --downloads C:\\temp\\dl`

Tools available to the Agent
These tool names are available for the LLM to call directly (ToolCallingAgent):

- navigate(url: string) -> string
  - Go to the given URL.

- click_element(selector_or_text: string, by?: "auto"|"text"|"link_text"|"css"|"xpath", nth?: int) -> string
  - Click an element by visible text, CSS, or XPATH. `nth` selects a specific match.

- fill_field(selector: string, text: string, by?: "auto"|"css"|"xpath", clear?: bool) -> string
  - Type into inputs by placeholder/label (helium) or by CSS/XPATH.

- submit_form(selector?: string) -> string
  - Submit focused form or click a submit button.

- scroll(num_pixels?: int=1200, direction?: "down"|"up") -> string
  - Scroll page.

- close_popups() -> string
  - Send ESCAPE to close modal overlays.

- login(url: string, username: string, password: string, username_selector?: string, password_selector?: string, submit_selector?: string) -> string
  - Visit login page, fill credentials, submit.

- find_on_page(text: string, nth_result?: int=1, max_context_chars?: int=400) -> string
  - Find occurrences of `text` on the page, scroll to nth match, and return a short snippet.
  - Helpful for large pages (e.g., big Wikipedia articles) to target the right section before sending content to the LLM.

- page_info() -> string
  - Returns URL, title, element count, HTML length, visible text length, and approx token estimate.
  - Use to decide whether to fetch the whole page or use `find_on_page` first.

- download_links(pattern?: string, max_files?: int=10, downloads_dir?: string) -> string
  - Download links from current page using current session cookies (auth-aware). If `pattern` is unset, filters by common extensions (images, video, pdf, zip, csv, json).

- extract_page_text() -> string
  - Extract visible text from the current page (truncates if very long).

- index_current_page() -> string
  - Index current page’s visible text into the per‑site RAG store and add discovered links to the site graph.

- build_site_tree(root_url?: string, max_pages?: int=50, max_depth?: int=2, same_domain_only?: bool=true) -> string
  - Small crawler from the current or given URL; indexes text and builds a limited site tree for later retrieval.

- get_site_tree(url?: string) -> string
  - Return a compact JSON of the site adjacency list (limited for readability).

- search_site_knowledge(query: string, k?: int=5) -> string
  - Search the per‑site TF‑IDF index and return relevant chunks (URL + snippet).

- create_and_run_script(filename: string, code: string, run?: bool=true) -> string
  - Write a Python script under `scripts/` and optionally execute it (returns stdout/stderr).

- python_interpreter(code: string) -> string
  - Evaluate Python code for math and quick data processing. Authorized imports: `math`, `statistics`, `json`, `re`, `itertools`, `collections`, `datetime`, `numpy`.

- web_search(query: string) -> string
  - Search the web via the default search tool.

Typical workflow for large pages
1) `page_info()` to assess size.
2) If large, `find_on_page("<keyword>")` to locate relevant sections.
3) Navigate or scroll near the match, then `extract_page_text()` or `index_current_page()`.
4) Use `search_site_knowledge()` to recall information later.

Notes
- Downloads use the same session as the browser (Selenium cookies are copied to the requests session).
- The per-site RAG uses a small TF‑IDF index stored under `site_index/` by domain.
- The agent automatically takes screenshots for each step; images are stored in memory for the run logs.

