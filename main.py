import os
import uuid
import asyncio
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from database import db, create_document
from schemas import Project, Asset, Page, PageBlock

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static uploads dir
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def read_root():
    return {"message": "Generative Publishing API running"}


# ------- Models for requests -------
class CreateProjectRequest(BaseModel):
    title: str
    prompt: str
    pages: int = 4
    preset: str = "modern"


@app.post("/api/projects")
async def create_project(payload: CreateProjectRequest):
    # Always create a project_id so the frontend can proceed, even if DB write fails
    project = Project(
        title=payload.title,
        prompt=payload.prompt,
        pages=max(1, min(64, payload.pages)),
        preset=payload.preset if payload.preset in [
            "minimal", "elegant", "bold", "modern", "real-estate"
        ] else "modern",
        status="created",
        assets=[],
        outline=[],
        preview_urls=[],
    )

    project_id = str(uuid.uuid4())
    mongo_id = None

    if db is not None:
        try:
            doc = project.model_dump()
            doc.update({"project_id": project_id})
            mongo_id = create_document("project", doc)
        except Exception:
            # Degrade gracefully if DB insert fails; continue without persistence
            mongo_id = None

    return {"project_id": project_id, "mongo_id": mongo_id, "status": "created"}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    if db is None:
        # Graceful fallback
        raise HTTPException(status_code=503, detail="Database not available")
    doc = db["project"].find_one({"project_id": project_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return doc


@app.post("/api/projects/{project_id}/assets")
async def upload_assets(project_id: str, files: List[UploadFile] = File(...)):
    if db is None:
        # Accept upload to disk for preview flows even if DB is unavailable
        saved_assets: List[dict] = []
        for f in files:
            file_id = f"{project_id}_{uuid.uuid4().hex}_{f.filename}"
            path = os.path.join(UPLOAD_DIR, file_id)
            content = await f.read()
            with open(path, "wb") as fp:
                fp.write(content)
            url = f"/uploads/{file_id}"
            asset = Asset(filename=f.filename, url=url, content_type=f.content_type, size=len(content)).model_dump()
            saved_assets.append(asset)
        return {"uploaded": len(saved_assets), "assets": saved_assets}

    proj = db["project"].find_one({"project_id": project_id})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    saved_assets: List[dict] = []
    for f in files:
        file_id = f"{project_id}_{uuid.uuid4().hex}_{f.filename}"
        path = os.path.join(UPLOAD_DIR, file_id)
        content = await f.read()
        with open(path, "wb") as fp:
            fp.write(content)
        url = f"/uploads/{file_id}"
        asset = Asset(filename=f.filename, url=url, content_type=f.content_type, size=len(content)).model_dump()
        saved_assets.append(asset)

    db["project"].update_one(
        {"project_id": project_id},
        {"$push": {"assets": {"$each": saved_assets}}, "$set": {"updated_at": asyncio.get_event_loop().time()}},
    )

    return {"uploaded": len(saved_assets), "assets": saved_assets}


# ------- WebSocket for generation -------

async def simulate_agentic_generation(ws: WebSocket, project_id: str):
    """Simulate a high-quality agentic pipeline and stream structured events.
    This updates the database status and sends granular updates suitable for a live preview.
    Works even if the database is not available (degraded mode).
    """
    proj = None
    if db is not None:
        try:
            proj = db["project"].find_one({"project_id": project_id})
        except Exception:
            proj = None

    # Fallback project if DB is unavailable or project not found
    if not proj:
        proj = {
            "title": "Design Preview",
            "pages": 4,
            "preset": "modern",
        }

    total_pages = int(proj.get("pages", 4))

    def set_status(status: str, extra: Optional[dict] = None):
        if db is None:
            return
        update = {"status": status}
        if extra:
            update.update(extra)
        try:
            db["project"].update_one({"project_id": project_id}, {"$set": update})
        except Exception:
            pass

    # Planning
    set_status("planning")
    await ws.send_json({"type": "status", "stage": "planning", "message": "Analyzing prompt and assets"})
    await asyncio.sleep(0.6)
    outline = [
        "Cover: Hero visual, bold headline, subtitle",
        "Editorial: Vision, value prop, highlights",
        "Feature Spread: Image-led composition with pull-quotes",
        "Details: Specs, amenities, or product breakdown",
        "Back Cover: Call-to-action and brand"
    ][: max(3, min(6, total_pages))]
    set_status("writing", {"outline": outline})
    await ws.send_json({"type": "outline", "items": outline})

    # Writing copy
    await ws.send_json({"type": "status", "stage": "writing", "message": "Authoring headlines and body copy"})
    await asyncio.sleep(0.6)

    # Layout per page
    set_status("layout")
    await ws.send_json({"type": "status", "stage": "layout", "message": "Composing grid and typographic system"})

    # Generate a clean geometric layout using normalized coordinates
    for page_num in range(1, total_pages + 1):
        blocks = []
        if page_num == 1:
            # Cover
            blocks.append(PageBlock(type="image", x=0, y=0, width=1, height=1, rotation=0, style={"fit": "cover", "opacity": 0.9}).model_dump())
            blocks.append(PageBlock(type="headline", x=0.08, y=0.12, width=0.8, height=0.18, style={"fontSize": 72, "weight": 800, "letterSpacing": -1.0}, content=proj.get("title", "Design Preview")).model_dump())
            blocks.append(PageBlock(type="text", x=0.08, y=0.32, width=0.6, height=0.12, style={"fontSize": 20, "opacity": 0.85}, content="Premium AI-crafted editorial layouts and image-led storytelling.").model_dump())
        else:
            # Inner page grid
            margin = 0.06
            col_w = (1 - margin * 2 - 0.04) / 2
            img_h = 0.48
            blocks.append(PageBlock(type="image", x=margin, y=margin, width=col_w, height=img_h, style={"radius": 16}).model_dump())
            blocks.append(PageBlock(type="text", x=margin + col_w + 0.04, y=margin, width=col_w, height=0.32, style={"fontSize": 16, "leading": 1.5}, content="Elegant, grid-driven layout with consistent rhythm and hierarchy.").model_dump())
            blocks.append(PageBlock(type="headline", x=margin + col_w + 0.04, y=margin + 0.36, width=col_w, height=0.12, style={"fontSize": 36, "weight": 700}, content="Section Title").model_dump())
            blocks.append(PageBlock(type="caption", x=margin, y=margin + img_h + 0.02, width=col_w, height=0.08, style={"fontSize": 12, "opacity": 0.7}, content="Fig. 1 — Visual narrative with thoughtful whitespace").model_dump())
        await ws.send_json({"type": "page", "page": page_num, "layout": blocks})
        await asyncio.sleep(0.4)

    # Rendering stage (could be rasterization/export in a real system)
    set_status("rendering")
    await ws.send_json({"type": "status", "stage": "rendering", "message": "Polishing kerning, exporting previews"})
    await asyncio.sleep(0.6)

    # Complete
    set_status("complete")
    await ws.send_json({"type": "complete", "message": "Project ready"})


@app.websocket("/ws/projects/{project_id}")
async def ws_generate(websocket: WebSocket, project_id: str):
    await websocket.accept()
    try:
        await simulate_agentic_generation(websocket, project_id)
    except WebSocketDisconnect:
        # Client disconnected; nothing else needed
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except RuntimeError:
            pass
        finally:
            try:
                if db is not None:
                    db["project"].update_one({"project_id": project_id}, {"$set": {"status": "error"}})
            except Exception:
                pass


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db as _db
        if _db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = _db.name if hasattr(_db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = _db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
