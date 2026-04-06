"""
RECALIX API — System Routes
Native file dialogs and system utilities for desktop app.
"""

import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["system"])


@router.post("/select-directory")
async def select_directory():
    """
    Open a native directory picker dialog.
    Returns the selected directory path or null if cancelled.
    """
    try:
        # Use tkinter for cross-platform native dialog
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Open directory picker
        directory = filedialog.askdirectory(
            title="Select Photo Directory",
            mustexist=True
        )
        
        root.destroy()
        
        return {"directory": directory if directory else None}
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="tkinter not available - cannot show native dialog"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dialog error: {str(e)}"
        )
