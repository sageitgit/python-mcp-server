from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

mcp = FastMCP("sageit-mcp-server")

@mcp.tool()
def hello(name: str = "world") -> dict:
    return {"message": f"Hello, {name}!"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK - MCP Server is running"

@app.get("/health")
def health():
    return {"status": "ok"}

# MCP over HTTP/SSE at /mcp
# path="/" helps avoid /mcp -> /mcp/ redirect issues in some setups
app.mount("/mcp", mcp.http_app(path="/"))
