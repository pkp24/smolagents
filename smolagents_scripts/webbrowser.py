import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

try:
    import helium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
except Exception as e:
    raise ImportError(
        "This script requires 'helium' and 'selenium'. Install them: pip install helium selenium"
    ) from e

try:
    import requests
    from bs4 import BeautifulSoup
except Exception as e:
    raise ImportError(
        "This script requires 'requests' and 'beautifulsoup4'. Install them: pip install requests beautifulsoup4"
    ) from e

from PIL import Image

from smolagents import (
    OpenAIServerModel,
    ToolCallingAgent,
    WebSearchTool,
    tool,
)
from smolagents.agents import ActionStep, CodeAgent
from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import Tool


# -------------------------
# Browser and environment
# -------------------------

GLOBAL_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
)

driver: webdriver.Chrome | None = None


def initialize_driver(headless: bool, downloads_dir: str) -> webdriver.Chrome:
    chrome_options = webdriver.ChromeOptions()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    # Prefer dark mode for both browser UI and web contents
    chrome_options.add_argument("--force-dark-mode")
    chrome_options.add_argument("--enable-features=WebUIDarkMode,WebContentsForceDark")
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1200,1400")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    chrome_options.add_argument(f"--user-agent={GLOBAL_USER_AGENT}")
    prefs = {
        "download.default_directory": os.path.abspath(downloads_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    return helium.start_chrome(headless=headless, options=chrome_options)


def get_driver_or_raise() -> webdriver.Chrome:
    if driver is None:
        raise RuntimeError("Browser driver not initialized.")
    return driver


# -------------------------
# Utilities
# -------------------------


def to_abs_url(base_url: str, href: str) -> str:
    from urllib.parse import urljoin

    return urljoin(base_url, href)


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return text


def get_requests_session_from_driver(current_driver: webdriver.Chrome) -> requests.Session:
    sess = requests.Session()
    # Copy cookies from Selenium to requests
    try:
        for c in current_driver.get_cookies():
            cookie_dict = {
                "domain": c.get("domain"),
                "name": c.get("name"),
                "value": c.get("value"),
                "path": c.get("path", "/"),
            }
            sess.cookies.set(**{k: v for k, v in cookie_dict.items() if v is not None})
    except Exception:
        pass
    sess.headers.update({"User-Agent": GLOBAL_USER_AGENT})
    return sess


# -------------------------
# Form utilities
# -------------------------

def _collect_labels_map(drv: webdriver.Chrome) -> Dict[str, str]:
    """
    Build a map from input id -> label text using <label for="...">.
    """
    labels: Dict[str, str] = {}
    try:
        for lbl in drv.find_elements(By.TAG_NAME, "label"):
            for_attr = (lbl.get_attribute("for") or "").strip()
            text = (lbl.text or "").strip()
            if for_attr and text:
                labels[for_attr] = text
    except Exception:
        pass
    return labels


def _nearest_label_text(drv: webdriver.Chrome, elem) -> str:
    """Try to find a nearby label text for a field without an explicit <label for> mapping."""
    try:
        # Check parent label
        parent = elem.find_element(By.XPATH, "ancestor::label[1]")
        if parent:
            t = (parent.text or "").strip()
            if t:
                return t
    except Exception:
        pass
    try:
        # Check previous sibling label
        prev_label = elem.find_element(By.XPATH, "preceding-sibling::label[1]")
        if prev_label:
            t = (prev_label.text or "").strip()
            if t:
                return t
    except Exception:
        pass
    return ""


def _find_field_by_label_like(drv: webdriver.Chrome, label_text: str):
    """
    Find an input/textarea/select whose label/placeholder/aria-label/name/id best matches label_text.
    """
    def _norm(s: str) -> str:
        return (s or "").strip().lower().rstrip(": ")

    label_text_l = _norm(label_text)
    labels_map = _collect_labels_map(drv)

    # 1) Exact match via label for=id
    for input_id, lbl in labels_map.items():
        if _norm(lbl) == label_text_l:
            try:
                return drv.find_element(By.ID, input_id)
            except Exception:
                pass

    # 2) If label mentions common field types, try direct ID/NAME hits
    token_hints: list[str] = []
    if "name" in label_text_l:
        token_hints.append("name")
    if "password" in label_text_l:
        token_hints.append("password")
    if "email" in label_text_l:
        token_hints.append("email")
    for tok in token_hints:
        try:
            el = drv.find_element(By.ID, tok)
            return el
        except Exception:
            try:
                el = drv.find_element(By.NAME, tok)
                return el
            except Exception:
                pass

    # 3) Scan all fields and score attributes
    candidates = drv.find_elements(By.XPATH, "//input|//textarea|//select")
    best = None
    best_score = 0
    for el in candidates:
        attrs = {
            "name": (el.get_attribute("name") or "").lower(),
            "id": (el.get_attribute("id") or "").lower(),
            "placeholder": (el.get_attribute("placeholder") or "").lower(),
            "aria_label": (el.get_attribute("aria-label") or "").lower(),
            "type": (el.get_attribute("type") or "").lower(),
        }
        near = _nearest_label_text(drv, el).lower()
        values = list(attrs.values()) + [near]
        score = max((1 if label_text_l in v and len(label_text_l) >= 2 else 0) for v in values) if values else 0
        if score > best_score:
            best_score = score
            best = el
    if best is not None:
        return best

    # 4) Look for adjacent text in tables or two-column layouts
    #    Example: <td>Your Name:</td><td><input id="name"></td>
    x_label = _xpath_literal(label_text_l)
    # try exact normalized text in td/th
    xpaths = [
        f"//td[translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')={x_label}]/following-sibling::td[1]//input|//td[translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')={x_label}]/following-sibling::td[1]//textarea|//td[translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')={x_label}]/following-sibling::td[1]//select",
        f"//tr[td and (translate(normalize-space(td[1]), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')={x_label})]//input|//tr[td and (translate(normalize-space(td[1]), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')={x_label})]//textarea|//tr[td and (translate(normalize-space(td[1]), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')={x_label})]//select",
    ]
    for xp in xpaths:
        try:
            found = drv.find_elements(By.XPATH, xp)
            if found:
                return found[0]
        except Exception:
            pass
    return None


# -------------------------
# Lightweight per-site RAG
# -------------------------


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class SiteChunk:
    url: str
    title: str
    text: str


class SiteIndexManager:
    def __init__(self, index_root: str = "site_index"):
        self.index_root = index_root
        os.makedirs(self.index_root, exist_ok=True)
        self.index_cache: Dict[str, Dict[str, Any]] = {}

    def _domain_key(self, url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc.replace(":", "_")

    def _index_path(self, domain_key: str) -> str:
        return os.path.join(self.index_root, f"{domain_key}.json")

    def load_domain(self, domain_key: str) -> Dict[str, Any]:
        if domain_key in self.index_cache:
            return self.index_cache[domain_key]
        path = self._index_path(domain_key)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"chunks": [], "graph": {}, "created": time.time()}
        self.index_cache[domain_key] = data
        return data

    def save_domain(self, domain_key: str) -> None:
        path = self._index_path(domain_key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.index_cache[domain_key], f, ensure_ascii=False, indent=2)

    def add_page(self, url: str, title: str, text: str) -> int:
        domain_key = self._domain_key(url)
        data = self.load_domain(domain_key)
        count_before = len(data["chunks"])
        for ch in chunk_text(text):
            data["chunks"].append({"url": url, "title": title, "text": ch})
        self.save_domain(domain_key)
        return len(data["chunks"]) - count_before

    def add_edge(self, src_url: str, dst_url: str) -> None:
        domain_key = self._domain_key(src_url)
        data = self.load_domain(domain_key)
        data["graph"].setdefault(src_url, [])
        if dst_url not in data["graph"][src_url]:
            data["graph"][src_url].append(dst_url)
        self.save_domain(domain_key)

    def build_tf_idf(self, domain_key: str) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        data = self.load_domain(domain_key)
        chunks = data.get("chunks", [])
        # term frequency per chunk
        tfs: List[Dict[str, float]] = []
        df: Dict[str, int] = defaultdict(int)
        for ch in chunks:
            tokens = tokenize(ch["text"])[:20000]
            tf: Dict[str, float] = defaultdict(float)
            for tok in tokens:
                tf[tok] += 1.0
            for tok in set(tokens):
                df[tok] += 1
            # l2 normalize
            norm = max(1e-8, sum(v * v for v in tf.values()) ** 0.5)
            tf = {k: v / norm for k, v in tf.items()}
            tfs.append(tf)
        # idf
        import math

        n = max(1, len(chunks))
        idf = {t: math.log((n + 1) / (dfc + 1)) + 1.0 for t, dfc in df.items()}
        return tfs, idf

    def search(self, domain_key: str, query: str, k: int = 5) -> List[Tuple[int, float]]:
        data = self.load_domain(domain_key)
        chunks = data.get("chunks", [])
        if not chunks:
            return []
        tfs, idf = self.build_tf_idf(domain_key)
        q_tokens = tokenize(query)
        q_tf: Dict[str, float] = defaultdict(float)
        for tok in q_tokens:
            q_tf[tok] += 1.0
        # l2 normalize q
        norm_q = max(1e-8, sum(v * v for v in q_tf.values()) ** 0.5)
        q_tf = {k: v / norm_q for k, v in q_tf.items()}
        # score = cosine with idf weights
        scores: List[Tuple[int, float]] = []
        for idx, doc_tf in enumerate(tfs):
            s = 0.0
            for tok, qv in q_tf.items():
                if tok in doc_tf:
                    w = idf.get(tok, 1.0)
                    s += qv * doc_tf[tok] * (w * w)
            scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


SITE_INDEX = SiteIndexManager()


# -------------------------
# Tools (navigation, login, forms, form discovery, find, download, site tree, rag, scripts)
# -------------------------


@tool
def navigate(url: str) -> str:
    """
    Navigate the browser to a given URL.

    Args:
        url: Absolute or relative URL to visit.
    """
    drv = get_driver_or_raise()
    helium.go_to(url)
    time.sleep(1.0)
    return f"Navigated to {drv.current_url}"


@tool
def click_element(selector_or_text: str, by: str | None = None, nth: int | None = 1) -> str:
    """
    Click an element on the page by visible text, CSS selector, or XPATH.

    Args:
        selector_or_text: Visible text (button/link) or a CSS/XPATH selector.
        by: One of ["auto", "text", "link_text", "css", "xpath"]. Defaults to auto.
        nth: If multiple matches, click the nth occurrence (1-based).
    """
    drv = get_driver_or_raise()
    mode = (by or "auto").lower()

    def selenium_click(elem):
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
        elem.click()

    if mode in ("auto", "text", "link_text"):
        try:
            helium.click(selector_or_text)
            return f"Clicked element with text: {selector_or_text}"
        except Exception:
            pass
    if mode in ("auto", "css"):
        try:
            elems = drv.find_elements(By.CSS_SELECTOR, selector_or_text)
            if not elems:
                raise LookupError("No element for CSS selector")
            selenium_click(elems[(nth or 1) - 1])
            return f"Clicked CSS element: {selector_or_text}"
        except Exception:
            pass
    if mode in ("auto", "xpath"):
        elems = drv.find_elements(By.XPATH, selector_or_text)
        if not elems:
            raise LookupError("No element for XPATH selector")
        selenium_click(elems[(nth or 1) - 1])
        return f"Clicked XPATH element: {selector_or_text}"
    raise LookupError("Element not found to click.")


@tool
def scroll(num_pixels: int = 1200, direction: str = "down") -> str:
    """
    Scroll the page up or down by a number of pixels.

    Args:
        num_pixels: Number of pixels to scroll.
        direction: "down" or "up".
    """
    drv = get_driver_or_raise()
    if direction.lower() == "down":
        helium.scroll_down(num_pixels)
    else:
        helium.scroll_up(num_pixels)
    return f"Scrolled {direction} by {num_pixels} pixels at {drv.current_url}"


@tool
def close_popups() -> str:
    """Send ESC to close modal/popups. Use when a modal blocks the screen."""
    drv = get_driver_or_raise()
    webdriver.ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    return "Attempted to close popup with ESCAPE."


def _xpath_literal(s: str) -> str:
    """
    Safely build an XPath literal for arbitrary string s.
    Uses concat() when both quote types are present.
    """
    if '"' not in s:
        return f'"{s}"'
    if "'" not in s:
        return f"'{s}'"
    parts = s.split('"')
    return "concat(" + ", '\"', ".join([f'"{p}"' for p in parts]) + ")"


@tool
def find_on_page(text: str, nth_result: int | None = 1, max_context_chars: int | None = 400) -> str:
    """
    Find occurrences of the given text on the current page, scroll to the nth occurrence, and highlight it.

    Args:
        text: The substring to search for on the page (case-sensitive).
        nth_result: Which match to focus on (1-based). Defaults to 1.
        max_context_chars: Max characters of element text to include in the result snippet.
    """
    drv = get_driver_or_raise()
    # Find elements whose string-value contains the text
    literal = _xpath_literal(text)
    xpath = f"//*[contains(., {literal})]"
    elems = drv.find_elements(By.XPATH, xpath)
    if not elems:
        return f"No matches found for '{text}'."
    index = max(1, int(nth_result or 1))
    if index > len(elems):
        return f"Found {len(elems)} matches for '{text}', but nth_result={index} is out of range."
    target = elems[index - 1]
    try:
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'});", target)
        drv.execute_script(
            "arguments[0].setAttribute('data-smo-highlight','1'); arguments[0].style.outline='2px solid #ff4d4f';",
            target,
        )
    except Exception:
        pass
    raw_text = target.text or ""
    snippet = raw_text[: (max_context_chars or 400)]
    return f"Found {len(elems)} matches; focused on {index}. Snippet: {snippet}"


@tool
def fill_field(selector: str, text: str, by: str | None = None, clear: bool | None = True) -> str:
    """
    Type text into a field located by CSS/XPATH or inferred.

    Args:
        selector: Visible label/placeholder text, or selector depending on 'by'.
        text: Text to type.
        by: One of "auto", "text", "label", "css", "xpath", "name", "id", "placeholder", "aria".
        clear: Whether to clear the field before typing.
    """
    drv = get_driver_or_raise()
    mode = (by or "auto").lower()
    elem = None
    # Auto: try helium by label/placeholder, then label-like search, then css/xpath
    if mode == "auto":
        try:
            helium.write(text, into=selector)
            return f"Filled field via helium: {selector}"
        except Exception:
            elem = _find_field_by_label_like(drv, selector)
    elif mode in ("text", "label"):
        elem = _find_field_by_label_like(drv, selector)
    elif mode == "name":
        try:
            elem = drv.find_element(By.NAME, selector)
        except Exception:
            elem = None
    elif mode == "id":
        try:
            elem = drv.find_element(By.ID, selector)
        except Exception:
            elem = None
    elif mode == "placeholder":
        try:
            elem = drv.find_element(By.XPATH, f"//*[@placeholder={_xpath_literal(selector)}]")
        except Exception:
            elem = None
    elif mode == "aria":
        try:
            elem = drv.find_element(By.XPATH, f"//*[@aria-label={_xpath_literal(selector)}]")
        except Exception:
            elem = None
    elif mode == "css":
        try:
            elem = drv.find_element(By.CSS_SELECTOR, selector)
        except Exception:
            elem = None
    elif mode == "xpath":
        try:
            elem = drv.find_element(By.XPATH, selector)
        except Exception:
            elem = None
    if elem is None:
        raise LookupError("Field not found")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
    if clear:
        try:
            elem.clear()
        except Exception:
            pass
    elem.send_keys(text)
    return f"Filled field via selenium: {selector}"


@tool
def submit_form(selector: str | None = None) -> str:
    """
    Submit the nearest form or click a submit button.

    Args:
        selector: Optional CSS/XPATH selector for a submit button.
    """
    drv = get_driver_or_raise()
    if selector:
        try:
            return click_element(selector, by="css")
        except Exception:
            return click_element(selector, by="xpath")
    # Try Enter key on focused element
    webdriver.ActionChains(drv).send_keys(Keys.ENTER).perform()
    return "Submitted form via Enter key."


@tool
def list_form_fields() -> str:
    """
    Scan the page and list detected form fields with attributes to help decide what to fill.

    Returns a JSON string with an array of objects: {type,name,id,placeholder,aria_label,label,required,css,xpath}.
    """
    drv = get_driver_or_raise()
    labels_map = _collect_labels_map(drv)
    fields = drv.find_elements(By.XPATH, "//input|//textarea|//select")
    out: List[Dict[str, Any]] = []
    for i, el in enumerate(fields):
        try:
            tag = el.tag_name
            ftype = el.get_attribute("type") or ("select" if tag == "select" else tag)
            name = el.get_attribute("name") or ""
            idv = el.get_attribute("id") or ""
            placeholder = el.get_attribute("placeholder") or ""
            aria = el.get_attribute("aria-label") or ""
            label = labels_map.get(idv, "") or _nearest_label_text(drv, el)
            required = bool(el.get_attribute("required"))
            css = None
            xpath = None
            try:
                xpath = drv.execute_script(
                    "function absoluteXPath(element){var comp=[];while(element&&element.nodeType===1){var sib=0;var ns=0;var sibs=element.parentNode?element.parentNode.childNodes:[];for(var i=0;i<sibs.length;i++){var s=sibs[i];if(s.nodeType===1&&s.nodeName===element.nodeName){ns++;if(s===element){sib=ns}}}comp.unshift(element.nodeName.toLowerCase()+'['+sib+']');element=element.parentNode}return '//' + comp.join('/')}; return absoluteXPath(arguments[0]);",
                    el,
                )
            except Exception:
                pass
            out.append(
                {
                    "tag": tag,
                    "type": ftype,
                    "name": name,
                    "id": idv,
                    "placeholder": placeholder,
                    "aria_label": aria,
                    "label": label,
                    "required": required,
                    "xpath": xpath,
                }
            )
        except Exception:
            continue
    import json as _json

    return _json.dumps(out, ensure_ascii=False, indent=2)


class MultiActionTool(Tool):
    name = "multi_action"
    description = (
        "Execute a small sequence of built-in browser tools in order. Use to combine simple actions"
        " like closing popups, scrolling, navigating, clicking, and filling fields without incurring"
        " additional planning steps."
    )
    inputs = {
        "actions": {
            "type": "array",
            "description": (
                "A list of action objects. Each object has a 'tool' name and 'args' dict."
                " Allowed tools: navigate, close_popups, scroll, click_element, fill_field, submit_form,"
                " find_on_page, page_info."
            ),
        }
    }
    output_type = "string"

    def forward(self, actions: list[dict]) -> str:
        assert isinstance(actions, list) and actions, "'actions' must be a non-empty list"
        results: list[str] = []
        for idx, action in enumerate(actions, start=1):
            try:
                tool_name = action.get("tool")
                args = action.get("args", {})
                if tool_name == "navigate":
                    results.append(navigate(**args))
                elif tool_name == "close_popups":
                    results.append(close_popups())
                elif tool_name == "scroll":
                    results.append(scroll(**args))
                elif tool_name == "click_element":
                    results.append(click_element(**args))
                elif tool_name == "fill_field":
                    results.append(fill_field(**args))
                elif tool_name == "submit_form":
                    results.append(submit_form(**args))
                elif tool_name == "find_on_page":
                    results.append(find_on_page(**args))
                elif tool_name == "page_info":
                    results.append(page_info())
                else:
                    results.append(f"Unsupported tool: {tool_name}")
            except Exception as e:
                results.append(f"Error in '{action}': {e}")
        return "\n".join(results)


# Instantiate class tool
multi_action = MultiActionTool()


@tool
def login(
    url: str,
    username: str,
    password: str,
    username_selector: str | None = None,
    password_selector: str | None = None,
    submit_selector: str | None = None,
) -> str:
    """
    Log into a site by visiting a login page, filling credentials, and submitting.

    Args:
        url: Login page URL.
        username: Username or email.
        password: Password.
        username_selector: Optional CSS/XPATH or placeholder text for username.
        password_selector: Optional CSS/XPATH or placeholder text for password.
        submit_selector: Optional CSS/XPATH for submit button.
    """
    navigate(url)
    time.sleep(1.0)
    candidates_user = [
        username_selector,
        "input[name='username']",
        "input[name='email']",
        "input[type='email']",
        "input#username",
        "input#email",
        "//input[@name='username']",
        "//input[@name='email']",
    ]
    candidates_pass = [
        password_selector,
        "input[name='password']",
        "input[type='password']",
        "input#password",
        "//input[@name='password']",
        "//input[@type='password']",
    ]
    # Try helium first using placeholders/labels
    if username_selector is None:
        try:
            helium.write(username, into="Email")
        except Exception:
            pass
    if password_selector is None:
        try:
            helium.write(password, into="Password")
        except Exception:
            pass
    # Selenium fallback
    for sel in [s for s in candidates_user if s]:
        try:
            fill_field(sel, username, by="css")
            break
        except Exception:
            try:
                fill_field(sel, username, by="xpath")
                break
            except Exception:
                continue
    for sel in [s for s in candidates_pass if s]:
        try:
            fill_field(sel, password, by="css", clear=True)
            break
        except Exception:
            try:
                fill_field(sel, password, by="xpath", clear=True)
                break
            except Exception:
                continue
    if submit_selector:
        try:
            click_element(submit_selector, by="css")
        except Exception:
            click_element(submit_selector, by="xpath")
    else:
        # common submit button attempts
        for text in ("Sign in", "Log in", "Login", "Submit"):
            try:
                helium.click(text)
                break
            except Exception:
                continue
        else:
            submit_form()
    time.sleep(2.0)
    return f"Post-login URL: {get_driver_or_raise().current_url}"


DOWNLOAD_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".webm", ".zip", ".7z", ".rar", ".pdf", ".csv", ".json"}


def _collect_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    hrefs: List[str] = []
    for a in soup.find_all("a", href=True):
        hrefs.append(to_abs_url(base_url, a["href"]))
    return list(dict.fromkeys(hrefs))


@tool
def download_links(pattern: str | None = None, max_files: int = 10, downloads_dir: str | None = None) -> str:
    """
    Download files linked on the current page. Uses browser cookies for authenticated downloads.

    Args:
        pattern: Optional regex to filter URLs or extensions. If None, common extensions are used.
        max_files: Maximum number of files to download.
        downloads_dir: Optional override for download directory.
    """
    drv = get_driver_or_raise()
    sess = get_requests_session_from_driver(drv)
    base = drv.current_url
    html = drv.page_source
    urls = _collect_links(html, base)

    dl_dir = os.path.abspath(downloads_dir or "downloads")
    os.makedirs(dl_dir, exist_ok=True)

    rx = re.compile(pattern) if pattern else None
    selected: List[str] = []
    for u in urls:
        if rx is not None:
            if rx.search(u):
                selected.append(u)
        else:
            from urllib.parse import urlparse
            ext = os.path.splitext(urlparse(u).path)[1].lower()
            if ext in DOWNLOAD_EXTS:
                selected.append(u)
        if len(selected) >= max_files:
            break

    saved = []
    for u in selected:
        try:
            r = sess.get(u, timeout=60, stream=True)
            r.raise_for_status()
            filename = None
            cd = r.headers.get("content-disposition", "")
            m = re.search(r'filename\*?=([^;]+)', cd, flags=re.IGNORECASE)
            if m:
                filename = m.group(1).strip().strip('"')
            if not filename:
                from urllib.parse import urlparse

                filename = os.path.basename(urlparse(u).path) or f"download_{len(saved)}"
            out_path = os.path.join(dl_dir, filename)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            saved.append(out_path)
        except Exception as e:
            saved.append(f"ERROR {u}: {e}")
    return "\n".join(saved) if saved else "No matching links to download."


@tool
def page_info() -> str:
    """
    Return page metadata: URL, title, total HTML length, visible text length, approximate token estimates, and element count.
    Use this to decide whether to read the whole page or target specific areas with find_on_page.
    """
    drv = get_driver_or_raise()
    html = drv.page_source
    text = extract_visible_text(html)
    try:
        elem_count = len(drv.find_elements(By.XPATH, '//*'))
    except Exception:
        elem_count = -1
    html_len = len(html)
    text_len = len(text)
    # Rough token estimate assuming ~4 chars/token
    approx_tokens = int((text_len / 4) + 0.5)
    return (
        f"URL: {drv.current_url}\nTitle: {drv.title}\nElements: {elem_count}\nHTML length: {html_len}\n"
        f"Visible text length: {text_len}\nApprox tokens: {approx_tokens}"
    )


@tool
def index_current_page() -> str:
    """
    Index the current page content into the per-site RAG store and update the site graph with discovered links.
    """
    drv = get_driver_or_raise()
    url = drv.current_url
    title = drv.title or ""
    html = drv.page_source
    text = extract_visible_text(html)
    added = SITE_INDEX.add_page(url, title, text)
    # add edges
    for link in _collect_links(html, url):
        SITE_INDEX.add_edge(url, link)
    return f"Indexed {added} chunk(s) for {url}."


@tool
def search_site_knowledge(query: str, k: int = 5) -> str:
    """
    Search the per-site RAG index for relevant chunks.

    Args:
        query: Search query.
        k: Max results.
    """
    drv = get_driver_or_raise()
    domain_key = SITE_INDEX._domain_key(drv.current_url)
    hits = SITE_INDEX.search(domain_key, query, k=k)
    data = SITE_INDEX.load_domain(domain_key)
    if not hits:
        return "No indexed content yet. Use index_current_page or build_site_tree first."
    outputs = []
    for idx, score in hits:
        ch = data["chunks"][idx]
        snippet = ch["text"][:800]
        outputs.append(f"- {ch['url']} | score={score:.3f}\n{snippet}")
    return "\n".join(outputs)


@tool
def build_site_tree(root_url: str | None = None, max_pages: int = 50, max_depth: int = 2, same_domain_only: bool = True) -> str:
    """
    Crawl links from the current page (or provided root_url) to build a limited site tree and index text for RAG.

    Args:
        root_url: Optional starting URL. Defaults to current page URL.
        max_pages: Crawl limit.
        max_depth: Max link depth.
        same_domain_only: Restrict to same domain.
    """
    drv = get_driver_or_raise()
    start_url = root_url or drv.current_url
    sess = get_requests_session_from_driver(drv)

    from urllib.parse import urlparse

    start_domain = urlparse(start_url).netloc
    seen = set()
    queue: List[Tuple[str, int]] = [(start_url, 0)]
    pages = 0
    while queue and pages < max_pages:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
            html = resp.text
            text = extract_visible_text(html)
            title = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
            title_text = (title.group(1).strip() if title else "").replace("\n", " ")
            added = SITE_INDEX.add_page(url, title_text, text)
            for link in _collect_links(html, url):
                SITE_INDEX.add_edge(url, link)
                if same_domain_only and urlparse(link).netloc != start_domain:
                    continue
                if link not in seen:
                    queue.append((link, depth + 1))
            pages += 1
        except Exception:
            continue
    return f"Crawled {pages} page(s) from {start_url}. Tree stored under site_index/."


@tool
def get_site_tree(url: str | None = None) -> str:
    """
    Return the stored adjacency list (limited site tree) as JSON for the site of the given or current URL.

    Args:
        url: The page URL whose site's tree should be returned. If not provided, uses the current page URL.
    """
    drv = get_driver_or_raise()
    domain_key = SITE_INDEX._domain_key(url or drv.current_url)
    data = SITE_INDEX.load_domain(domain_key)
    # Limit output for readability
    small_graph = {k: v[:10] for k, v in list(data.get("graph", {}).items())[:50]}
    return json.dumps(small_graph, ensure_ascii=False, indent=2)


@tool
def extract_page_text() -> str:
    """Extract visible text content from the current page and return it (truncated if very long)."""
    drv = get_driver_or_raise()
    text = extract_visible_text(drv.page_source)
    if len(text) > 20000:
        return text[:10000] + "\n...\n" + text[-10000:]
    return text


@tool
def create_and_run_script(filename: str, code: str, run: bool | None = True) -> str:
    """
    Create a Python script on disk and optionally execute it, returning stdout/stderr.

    Args:
        filename: The script filename (e.g., "process_data.py"). Will be saved under scripts/.
        code: Full Python source code to write.
        run: If true, execute after writing.
    """
    os.makedirs("scripts", exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    path = os.path.join("scripts", safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    if not run:
        return f"Wrote script at {path} (not executed)."
    # Execute in a subprocess
    import subprocess

    try:
        out = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=600)
        return f"Exit={out.returncode}\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    except Exception as e:
        return f"Execution error: {e}"


# -------------------------
# Agent, screenshots, and instructions
# -------------------------


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    time.sleep(1.0)
    drv = get_driver_or_raise()
    current_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:
        if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
            previous_memory_step.observations_images = None
    png_bytes = drv.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    memory_step.observations_images = [image.copy()]
    url_info = f"Current url: {drv.current_url}"
    memory_step.observations = url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info


INSTRUCTIONS = (
    "Use web_search for search queries. Use navigate/click_element/fill_field/submit_form/scroll to interact with pages. "
    "Use login to authenticate when needed. Use index_current_page and build_site_tree to feed the site RAG, then search_site_knowledge "
    "to find information later. Use download_links to fetch files using current session. Use extract_page_text when you need raw text. "
    "For math or ad-hoc processing, prefer python_interpreter; for larger jobs, create_and_run_script. After you complete the user task, return a final answer."
)


def build_tools(downloads_dir: str) -> List[Any]:
    # Allow math with Python
    python_tool = PythonInterpreterTool(authorized_imports=[
        "math",
        "statistics",
        "json",
        "re",
        "itertools",
        "collections",
        "datetime",
        "numpy",
    ])
    # Bind downloads_dir default for download_links via partial-like wrapper
    # Tool decorator requires original signature; we keep downloads_dir optional but set env var instead
    os.environ.setdefault("WEBBROWSER_DOWNLOADS_DIR", os.path.abspath(downloads_dir))
    return [
        WebSearchTool(),
        navigate,
        click_element,
        fill_field,
        submit_form,
        scroll,
        close_popups,
        login,
        download_links,
        extract_page_text,
        index_current_page,
        build_site_tree,
        get_site_tree,
        search_site_knowledge,
        multi_action,
        create_and_run_script,
        python_tool,
    ]


def create_model_from_env(model_id: str | None, api_base: str | None) -> OpenAIServerModel:
    # Defaults aligned with myapi.md (Qwen/Qwen3-4B-Thinking-2507 + http://localhost:8000/v1)
    mid = model_id or os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Thinking-2507")
    base = api_base or os.getenv("API_BASE") or f"http://localhost:{os.getenv('PORT', '8000')}/v1"
    # OpenAI SDK requires an api_key set even if the server does not enforce auth.
    # Provide a harmless default if env var is not set.
    api_key = os.getenv("OPENAI_API_KEY", "no-key-required")
    return OpenAIServerModel(model_id=mid, api_base=base, api_key=api_key)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smolagents Web Browser Agent (Helium + per-site RAG)")
    p.add_argument("prompt", type=str, nargs="?", default="Go to example.com and summarize the homepage.")
    p.add_argument("--model-id", dest="model_id", type=str, default=None)
    p.add_argument("--api-base", dest="api_base", type=str, default=None)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--downloads", dest="downloads", type=str, default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=20)
    p.add_argument("--verbosity", dest="verbosity", type=int, default=2)
    # Screenshots attach images to messages; keep off for non‑multimodal models
    p.add_argument("--screenshots", action="store_true", default=False)
    return p.parse_args()


def run_agent(
    prompt: str,
    model_id: str | None,
    api_base: str | None,
    headless: bool,
    downloads_dir: str | None,
    max_steps: int,
    verbosity: int,
    enable_screenshots: bool,
) -> str:
    global driver
    load_dotenv()
    model = create_model_from_env(model_id, api_base)
    # Default downloads directory to this script folder if not provided
    if not downloads_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        downloads_dir = os.path.join(script_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    driver = initialize_driver(headless=headless, downloads_dir=downloads_dir)
    tools = build_tools(downloads_dir)

    # Use ToolCallingAgent so the model can pick tools
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity,
        step_callbacks=[save_screenshot] if enable_screenshots else [],
        planning_interval=4,
    )
    # Give helium import so it can write code snippets if needed
    if hasattr(agent, "python_executor"):
        agent.python_executor("from helium import *")

    try:
        return agent.run(prompt + "\n\n" + INSTRUCTIONS)
    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    out = run_agent(
        prompt=args.prompt,
        model_id=args.model_id,
        api_base=args.api_base,
        headless=args.headless,
        downloads_dir=args.downloads,
        max_steps=args.max_steps,
        verbosity=args.verbosity,
        enable_screenshots=args.screenshots,
    )
    print("\n=== FINAL OUTPUT ===\n")
    print(out)


if __name__ == "__main__":
    main()


