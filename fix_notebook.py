import re
import json
import os
from typing import List, Dict, Any

def extract_cells_from_vscode_notebook(file_path: str) -> List[Dict[str, Any]]:
    """Extract cells from VS Code XML notebook format"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all VSCode.Cell blocks
    cell_pattern = r'<VSCode\.Cell\s+id="([^"]*?)"\s+language="([^"]*?)"[^>]*?>(.*?)(?=<VSCode\.Cell|$)'
    matches = re.findall(cell_pattern, content, re.DOTALL)
    
    jupyter_cells = []
    
    for i, (cell_id, language, cell_content) in enumerate(matches):
        cell_content = cell_content.strip()
        
        # Remove </VSCode.Cell> if it exists
        if cell_content.endswith('</VSCode.Cell>'):
            cell_content = cell_content[:-len('</VSCode.Cell>')].strip()
        
        # Split content into lines
        lines = cell_content.split('\n')
        source_lines = []
        
        for j, line in enumerate(lines):
            if j == len(lines) - 1:  # Last line
                source_lines.append(line)
            else:
                source_lines.append(line + '\n')
        
        if language == 'markdown':
            jupyter_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines
            }
        else:  # Treat everything else as code (python, etc.)
            jupyter_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            }
        
        jupyter_cells.append(jupyter_cell)
    
    return jupyter_cells

def create_jupyter_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a standard Jupyter notebook structure"""
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def convert_notebook(input_file: str, output_file: str):
    """Convert VS Code notebook to standard Jupyter format"""
    
    print(f"🔄 Converting {input_file}...")
    
    # Extract cells from VS Code format
    cells = extract_cells_from_vscode_notebook(input_file)
    
    if not cells:
        print("❌ No cells found! Checking alternative extraction method...")
        
        # Alternative method - try to extract content differently
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for cell markers
        if '<VSCode.Cell' in content:
            print("🔍 VS Code format detected, but cells not extracting properly")
            # Try a simpler split approach
            parts = content.split('<VSCode.Cell')
            cells = []
            
            for part in parts[1:]:  # Skip first empty part
                if 'language=' in part:
                    # Extract language
                    lang_match = re.search(r'language="([^"]*)"', part)
                    language = lang_match.group(1) if lang_match else 'python'
                    
                    # Extract content (everything after the > until next cell or end)
                    content_start = part.find('>') + 1
                    if content_start > 0:
                        cell_content = part[content_start:].strip()
                        
                        # Remove </VSCode.Cell> if present
                        if '</VSCode.Cell>' in cell_content:
                            cell_content = cell_content.split('</VSCode.Cell>')[0].strip()
                        
                        lines = cell_content.split('\n')
                        source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                        
                        if language == 'markdown':
                            cell = {
                                "cell_type": "markdown",
                                "metadata": {},
                                "source": source
                            }
                        else:
                            cell = {
                                "cell_type": "code",
                                "execution_count": None,
                                "metadata": {},
                                "outputs": [],
                                "source": source
                            }
                        
                        cells.append(cell)
    
    if not cells:
        print("❌ Still no cells found. The file might already be in JSON format or corrupted.")
        return False
    
    # Create Jupyter notebook
    notebook = create_jupyter_notebook(cells)
    
    # Write the converted notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully converted notebook!")
    print(f"📊 Cells converted: {len(cells)}")
    print(f"💾 Output file: {output_file}")
    
    return True

if __name__ == "__main__":
    # Convert the problematic Kaggle notebook
    input_file = "malaria-cell-detection-deep-learning (1).ipynb"
    output_file = "malaria-cell-detection-deep-learning.ipynb"
    
    success = convert_notebook(input_file, output_file)
    
    if success:
        print(f"\n🎉 Conversion complete! The notebook should now render properly on GitHub.")
    else:
        print(f"\n❌ Conversion failed. Please check the input file format.")
