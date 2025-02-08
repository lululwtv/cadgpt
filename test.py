import nbformat as nbf

code_response = """```python
import cadquery as cq

# Parameters for the hexagonal nut
hex_diameter = 0.25  # Diameter of the inscribed circle
nut_height = 0.125   # Height of the nut

# Create a hexagonal nut
result = (
    cq.Workplane("XY")
    .polygon(6, hex_diameter)
    .extrude(nut_height)
    .faces("<Z")
    .workplane()
    .hole(hex_diameter / 2)
)
```"""
notebook_filename = "result.ipynb"

code_response_py = code_response.replace("```python","").replace("```","").strip()
with open(notebook_filename, "r") as f:
            nb = nbf.read(f, as_version=4)
new_code = code_response_py+"\ndisplay(result)"
new_code_cell = nbf.v4.new_code_cell(new_code)
if "id" in new_code_cell:
    del new_code_cell["id"]
nb.cells.append(new_code_cell)
with open(notebook_filename, "w") as f:
    nbf.write(nb, f)
