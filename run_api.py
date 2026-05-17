"""
Entry point — run with: python run_api.py
On Render, the $PORT environment variable is injected automatically.
Locally, defaults to port 8000.
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
