import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from gamesense.api.main import app as api

REDOC_JS = "https://cdn.jsdelivr.net/npm/redoc@2.5.4/bundles/redoc.standalone.js"
LOCAL_SERVER = "http://localhost:8000"

# Matches the colours in .streamlit/config.toml.
BACKGROUND, PANEL, BORDER, TEXT, MUTED, ACCENT = "#1b2838", "#16202d", "#2a3f5a", "#e6edf3", "#9fb3c8", "#66c0f4"
SUCCESS, ERROR = "#7ee2a8", "#f28b82"
FONT = '-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
REDOC_OPTIONS = {
    "expandResponses": "200",
    "hideDownloadButton": True,
    "theme": {
        "colors": {
            "primary": {"main": ACCENT},
            "text": {"primary": TEXT, "secondary": MUTED},
            "border": {"dark": BORDER, "light": BORDER},
            "responses": {
                "success": {"color": SUCCESS, "backgroundColor": "rgba(126, 226, 168, 0.08)", "tabTextColor": SUCCESS},
                "error": {"color": ERROR, "backgroundColor": "rgba(242, 139, 130, 0.08)", "tabTextColor": ERROR},
            },
        },
        "typography": {
            "fontFamily": FONT,
            "headings": {"fontFamily": FONT},
            "code": {"backgroundColor": BORDER, "color": TEXT},
            "links": {"color": ACCENT},
        },
        "sidebar": {"backgroundColor": PANEL, "textColor": TEXT},
        "rightPanel": {"backgroundColor": PANEL, "textColor": TEXT},
        "schema": {"nestedBackground": PANEL, "typeNameColor": MUTED, "linesColor": BORDER, "requireLabelColor": ERROR},
    },
}

# ReDoc's theme options do not reach these parts, so they are set directly.
REDOC_CSS = f"""
body {{ margin: 0; background: {BACKGROUND}; }}
#redoc table tr, #redoc table th, #redoc table td {{ background: transparent !important; color: {TEXT}; }}
#redoc h5, #redoc h5 span {{ color: {MUTED} !important; }}
#redoc .react-tabs__tab {{ background: {BORDER} !important; color: {TEXT} !important; }}
#redoc .react-tabs__tab--selected {{ background: {TEXT} !important; color: {BACKGROUND} !important; }}
#redoc select, #redoc option {{ background: {PANEL}; color: {TEXT}; }}
#redoc button + div {{ background: {PANEL} !important; }}
#redoc button + div * {{ background: transparent !important; color: {TEXT} !important; }}
"""


@st.cache_data
def openapi_spec() -> dict[str, Any]:
    """The API's own spec, with addresses pointed at a locally started API instead of this page's frame."""
    spec = api.openapi()
    info = spec["info"] | {"description": spec["info"]["description"].replace("(/docs)", f"({LOCAL_SERVER}/docs)")}
    return spec | {"info": info, "servers": [{"url": LOCAL_SERVER, "description": "Started from the repository"}]}


def script_json(value: Any) -> str:
    """JSON that is safe to place inside a <script> tag."""
    return json.dumps(value).replace("</", "<\\/")


st.title("API reference")
st.markdown(
    "GameSense also runs as an HTTP API built with FastAPI. It uses the same model and input validation as this "
    f"app. Start it with `docker compose up` or `uvicorn gamesense.api.main:app` and open {LOCAL_SERVER}. "
    "The reference below is generated from the API's code."
)

components.html(
    f"""
    <style>{REDOC_CSS}</style>
    <div id="redoc"></div>
    <script src="{REDOC_JS}"></script>
    <script>
      Redoc.init({script_json(openapi_spec())}, {script_json(REDOC_OPTIONS)}, document.getElementById("redoc"));
      // Open links to other sites in a new tab instead of inside this frame.
      document.addEventListener("click", (event) => {{
        const link = event.target.closest("a[href^='http']");
        if (link) {{
          event.preventDefault();
          window.open(link.href, "_blank", "noopener");
        }}
      }});
    </script>
    """,
    height=1100,
    scrolling=True,
)
