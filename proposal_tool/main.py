from mcp.server.fastmcp import FastMCP
import json
import sys
import os
import warnings
import logging
import tempfile
import webbrowser
from datetime import date
from bs4 import BeautifulSoup, Tag, NavigableString
import re
from rapidfuzz import fuzz

# Silence noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logs.data_logging import data_logger
from views import template
from systems.pdf_tools import html_to_pdf
from systems.llm_config import proposal_change


log=data_logger()

mcp = FastMCP("Create Proposal")

def parse_html(html_content):
    return BeautifulSoup(html_content, "html.parser")

def extract_candidate_blocks(soup, tags=("tr","div","section","footer","header","p","span","td")):
    candidates = []
    for tag in soup.find_all(tags):
        text = tag.get_text(" ", strip=True)
        html = str(tag)
        if text:
            candidates.append({"text": text, "html": html, "tag": tag})
    return candidates

def find_target_block(prompt, candidates, threshold=50):
    best_score = 0
    best_block = None
    for c in candidates:
        score = fuzz.partial_ratio(prompt.lower(), c["text"].lower())
        if score > best_score:
            best_score = score
            best_block = c

    if best_block and best_score >= threshold:
        tag = best_block["tag"]
        if tag.name == "td" and tag.parent and tag.parent.name == "tr":
            return {"text": tag.parent.get_text(" ", strip=True), "html": str(tag.parent), "tag": tag.parent}
        return best_block
    return None

def save_updated_html(rfp_id, html_content):
    proposal_temp_path = os.path.join(tempfile.gettempdir(), "proposal.html")
    with open(proposal_temp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return proposal_temp_path

@mcp.tool(description="""
Display the proposal when user asks to display or prepare a proposal.
""")
async def display_proposal(rfp_id: str) -> str:
    """Render and display a saved proposal (from logs) as HTML and PDF."""

    try:
        # === Load JSON from logs ===
        proposal_data = log._load_logs()[rfp_id]["tools"]["quotation"]["result"]["updated_result_json"]

        # === Render HTML ===
        proposal_html = template.render_proposal(
            proposal_data,
            today=date.today().strftime("%m/%d/%Y"),
            basic=proposal_data[list(proposal_data.keys())[0]]
        )
        await html_to_pdf(proposal_html['template'], rfp_id, "proposal.pdf")

    except Exception as e:
        return f"❌ Proposal rendering failed: {e}"

    # === Save to shared html_content.json ===
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(path, "html_content.json")

    content = {"rfp_id": rfp_id, "proposal_html": proposal_html['template'], "proposal_json": proposal_data}
    with open(file_path, "w") as f:
        json.dump(content, f, indent=4)

    # === Preview in browser ===
    try:
        proposal_temp_path = os.path.join(tempfile.gettempdir(), f"proposal.html")
        with open(proposal_temp_path, "w", encoding="utf-8") as f:
            f.write(proposal_html['template'])

        # Open only once per rfp_id
        if not getattr(display_proposal, "opened", {}):
            display_proposal.opened = {}
        if rfp_id not in display_proposal.opened:
            webbrowser.open(f"file://{proposal_temp_path}")
            display_proposal.opened[rfp_id] = True
        log.log_proposal(rfp_id, {
            "proposal_html": proposal_html['template'],
            "json_data": proposal_html['data'],
            "updated_proposal_html": proposal_html['template']
        })

    except Exception as e:
        print(f"⚠️ Preview failed: {e}")

    return f"✅ Proposal displayed successfully for RFP {rfp_id}."
def detect_action(prompt: str) -> str:
    pl = (prompt or "").lower()
    if pl.startswith("add") or " add " in pl or "append" in pl:
        return "add"
    if any(x in pl for x in ["remove", "delete"]):
        return "remove"
    # update patterns like "change X to Y" / "update X to Y"
    if re.search(r'\b(change|update|replace)\b', pl):
        # if it contains "to" or "as" it's likely an update with new value
        if re.search(r'\b(to|as|=)\b', pl):
            return "update"
        return "update"
    return "update"

def get_edit_target(tag: Tag) -> Tag:
    """
    Expand the matched element into the right editable block.

    - If matched tag is <td> -> return <tr>
    - If matched tag is <li> -> return that <li> (caller can decide)
    - If matched tag is a title DIV/H2/H3/H4 and next sibling is UL/OL -> return the UL/OL
    - If matched tag itself is UL/OL -> return it
    - Otherwise return the tag itself
    """
    if not isinstance(tag, Tag):
        return tag

    # If inside a table cell -> bubble to the row
    if tag.name == "td" and tag.parent and getattr(tag.parent, "name", "") == "tr":
        return tag.parent

    # If tag is span inside a td -> try to bubble up
    if tag.name == "span" and tag.parent and getattr(tag.parent, "name", "") == "td":
        tr = tag.parent.parent
        if tr and getattr(tr, "name", "") == "tr":
            return tr
        return tag.parent

    # If tag is li -> we operate on the list (return parent) OR allow li-specific ops
    if tag.name == "li":
        return tag  # caller can choose to operate on parent list or this li

    # If tag already a list -> return it directly
    if tag.name in ("ul", "ol"):
        return tag

    # If it's a section heading / title, check immediate next siblings for a list
    if tag.name in ("div", "h2", "h3", "h4", "h1"):
        # skip if tag is itself a list-like header but not a textual header? we treat generically
        # look at immediate next sibling(s), skipping whitespace/text nodes
        nxt = tag.find_next_sibling()
        while nxt and (isinstance(nxt, NavigableString) or (getattr(nxt, "name", None) is None)):
            nxt = nxt.find_next_sibling()
        if nxt and getattr(nxt, "name", "") in ("ul", "ol"):
            return nxt
        # sometimes list is wrapped in a div; check within next few siblings for nearest ul/ol before next header/div.section
        cursor = tag.find_next_sibling()
        steps = 0
        while cursor and steps < 6:
            if isinstance(cursor, Tag) and cursor.name in ("ul", "ol"):
                return cursor
            # stop if we hit another major section header
            if isinstance(cursor, Tag) and cursor.name in ("div", "section", "header", "footer", "h2", "h3", "h4"):
                break
            cursor = cursor.find_next_sibling()
            steps += 1

    # default: return tag itself
    return tag

def locate_section_block(prompt: str, soup: BeautifulSoup, fuzz_threshold: int = 55) -> Tag | None:
    """
    Locate the container block (div/section/article) in HTML that best matches
    the user's prompt. Works for ANY section (not hard-coded keywords).

    Args:
        prompt: Natural language query
        soup: BeautifulSoup object of the HTML
        fuzz_threshold: minimum fuzzy score to accept a heading match

    Returns:
        BeautifulSoup Tag (div/section/article/etc.) or None
    """
    p = (prompt or "").lower()

    # 1) Fuzzy match prompt against headings
    best_score = 0
    best_heading = None
    for heading in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
        htext = heading.get_text(" ", strip=True).lower()
        score = fuzz.partial_ratio(p, htext)
        if score > best_score:
            best_score = score
            best_heading = heading

    if best_heading and best_score >= fuzz_threshold:
        # Climb upward to find the logical container
        for anc in best_heading.parents:
            if getattr(anc, "name", None) in ("div", "section", "article", "body"):
                # Prefer ancestor containing table/ul/ol (structured block)
                if anc.find(["table","ul","ol"]):
                    return anc
                return anc
        return best_heading

    # 2) If no heading matched, fuzzy match all candidate blocks
    candidates = extract_candidate_blocks(
        soup,
        tags=("div","section","article","tr","p","td","span")
    )
    flat_match = find_target_block(prompt, candidates, threshold=fuzz_threshold)
    if flat_match:
        return flat_match["tag"]

    return None

def extract_sections(soup):
    sections = []
    for heading in soup.find_all(["h1","h2","h3","h4","strong","b"]):
        heading_text = heading.get_text(" ", strip=True)
        # find next content (list, table, or paragraph)
        nxt = heading.find_next_sibling()
        while nxt and (nxt.name is None or nxt.get_text(strip=True) == ""):
            nxt = nxt.find_next_sibling()
        if nxt:
            sections.append({"heading": heading_text, "content": nxt, "html": str(nxt)})
    return sections

@mcp.tool(description="""after creating proposal when user ask to make chaneges in the proposal call this tool with rfp_id and user query
        user_queries is a list of strings containing the changes to be made in the proposal""")
def make_changes_in_proposal(rfp_id: str, user_queries: list) -> str:
    """
    For each query: locate the whole container block (div/section) for the section named
    in the prompt, send that block + prompt to LLM, get updated block back, replace it
    in the original HTML, and print the final HTML at the end process.
    """
    # load current proposal HTML from logs (same as your original)
    result = log._load_logs()[rfp_id]["tools"]["proposal"]["result"]
    html_content = result["updated_proposal_html"]

    for user_query in user_queries:
        soup = parse_html(html_content)

        # locate the block Tag (NOT string)
        edit_target = locate_section_block(user_query, soup)

        if not edit_target:
            print("⚠️ No matching section block found for:", user_query)
            # fallback to old behaviour: try find_target_block
            candidates = extract_candidate_blocks(soup)
            flat = find_target_block(user_query, candidates)
            if flat:
                edit_target = get_edit_target(flat["tag"])
            else:
                print("❌ No fallback match either. Skipping.")
                continue

        # ensure tag
        if not isinstance(edit_target, Tag):
            print("⚠️ locate_section_block returned non-Tag. Skipping.")
            continue

        # preview
        preview = edit_target.get_text(" ", strip=True)[:300].replace("\n", " ")
        print(f"\n🔎 Matched Block (tag: <{edit_target.name}>): {preview}")

        action = detect_action(user_query)

        # send the block + query to the LLM handler
        try:
            block_html = str(edit_target)
            updated_html = proposal_change(user_query, block_html, action)  # your LLM function
            if not updated_html:
                print("⚠️ LLM returned empty response for query:", user_query)
                continue

            # Parse LLM result and decide how to replace
            new_doc = BeautifulSoup(updated_html, "html.parser")

            # find first non-empty tag in the returned result
            first_tag = None
            for node in new_doc.contents:
                if isinstance(node, Tag):
                    first_tag = node
                    break
                # if text node with content, keep as fallback
                if isinstance(node, NavigableString) and node.strip():
                    # wrap text later
                    break

            # Replacement logic:
            # If first_tag exists and matches edit_target tag -> replace
            if first_tag and getattr(first_tag, "name", None) == edit_target.name:
                # create a replacement tag parsed by the original soup to avoid cross-soup issues
                replacement = BeautifulSoup(str(first_tag), "html.parser").find(True)
                edit_target.replace_with(replacement)

            else:
                # If the original block contains a table and LLM returned a table/tbody/tr rows
                orig_table = edit_target.find("table")
                # search for table/tbody/tr in new_doc
                new_table = new_doc.find("table")
                new_tbody = new_doc.find("tbody")
                new_trs = new_doc.find_all("tr")

                if orig_table and (new_table or new_tbody or new_trs):
                    if new_table:
                        replacement_table = BeautifulSoup(str(new_table), "html.parser").find("table")
                        orig_table.replace_with(replacement_table)
                    elif new_tbody:
                        replacement_tbody = BeautifulSoup(str(new_tbody), "html.parser").find("tbody")
                        existing_tbody = orig_table.find("tbody")
                        if existing_tbody:
                            existing_tbody.replace_with(replacement_tbody)
                        else:
                            orig_table.append(replacement_tbody)
                    elif new_trs:
                        # append each returned <tr> into existing table's tbody or table
                        tbody = orig_table.find("tbody") or orig_table
                        for tr in new_trs:
                            # ensure we insert soup-created tags to avoid cross-soup problems
                            replacement_row = BeautifulSoup(str(tr), "html.parser").find("tr")
                            tbody.append(replacement_row)
                    else:
                        # fallback: replace the whole block
                        replacement = BeautifulSoup(str(new_doc), "html.parser")
                        edit_target.replace_with(replacement)
                else:
                    # final fallback: if LLM returned several nodes, wrap them and replace the whole block
                    # or if only text - replace content
                    # build wrapper from the returned HTML and replace
                    replacement_wrapper = BeautifulSoup(str(new_doc), "html.parser")
                    # if replacement_wrapper has a top-level tag, use that
                    top_tag = replacement_wrapper.find(True)
                    if top_tag:
                        edit_target.replace_with(top_tag)
                    else:
                        # plain text - set the inner text of the edit_target
                        edit_target.string = updated_html

            # write changes back to html_content for next iteration
            html_content = str(soup)
            print("🔄 LLM update applied successfully for query:", user_query)

        except Exception as e:
            print("❌ proposal_change / replacement failed; error:", e)
            continue

    # Save back to logs after all edits
    result["updated_proposal_html"] = html_content
    log.log_proposal(rfp_id=rfp_id, result=result)

    # Save file for preview and printing (your existing helper)
    save_updated_html(rfp_id, html_content)

    # Print the full updated HTML (as you requested)

    return "finished"
    
# ===== START SERVER =====
if __name__ == "__main__":
    try:
        import asyncio
        print("✅ Starting MCP Server...", file=sys.stderr)
        mcp.run()
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}", file=sys.stderr)