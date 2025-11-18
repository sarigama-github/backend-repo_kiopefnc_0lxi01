"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any

# Example schemas (retain for reference)
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# --------------------
# Generative Publishing SaaS Schemas
# --------------------

DesignPreset = Literal[
    "minimal",
    "elegant",
    "bold",
    "modern",
    "real-estate",
]

class Asset(BaseModel):
    filename: str
    url: str
    content_type: Optional[str] = None
    size: Optional[int] = None

class PageBlock(BaseModel):
    type: Literal["image", "text", "headline", "caption", "shape"]
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0
    style: Dict[str, Any] = {}
    content: Optional[str] = None
    asset_url: Optional[str] = None

class Page(BaseModel):
    number: int
    layout: List[PageBlock] = []

class Project(BaseModel):
    title: str
    prompt: str
    pages: int = Field(4, ge=1, le=64)
    preset: DesignPreset = "modern"
    status: Literal["created", "planning", "writing", "layout", "rendering", "complete", "error"] = "created"
    assets: List[Asset] = []
    outline: Optional[List[str]] = None
    preview_urls: Optional[List[str]] = None
