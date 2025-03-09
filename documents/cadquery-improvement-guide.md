# CADQuery Code Generation Guidelines

## Common Issues in CAD Generation

1. **Wire Projection Problems**: Projecting wires onto curved surfaces is complex and error-prone.
   - The original code created a hexagon wire and attempted to project it onto the cylinder surface
   - This approach often fails with "no pending wires present" errors

2. **Incomplete Cut Operations**: Cutting operations need properly positioned and sized solid objects.
   - The original code created projected wires but didn't properly use them for cutting
   - It attempted to cut with an empty workplane (`tube.cut(cq.Workplane("XY").pushPoints([(0, 0)]).extrude(height))`)

3. **Parameter Clarity**: CAD operations need explicit, well-defined parameters.
   - The original code had unclear relationships between parameters (e.g., hex_diagonal calculation)
   - Wall thickness wasn't explicitly defined

## Improved Approach Guidelines

1. **Use Direct 3D Positioning**
   - Instead of complex wire projections, position workplanes directly in 3D space
   - Use trigonometric functions to calculate positions on curved surfaces
   - Example: `x = radius * cos(angle * pi / 180); y = radius * sin(angle * pi / 180)`

2. **Create Proper Cutting Solids**
   - Generate complete solid objects for boolean operations
   - For cuts through walls, ensure the cutting solid extends fully through the material
   - Example: `hole_plane.polygon(6, hex_size).extrude(wall_thickness * 2, both=True)`

3. **Use Transformed Workplanes**
   - Position and orient workplanes to align with the desired cutting direction
   - Use the `.transformed()` method with proper offset and rotation parameters
   - Example: `cq.Workplane("XY").transformed(offset=(x, y, z), rotate=(0, 90, angle))`

4. **Organize with Nested Loops**
   - Use nested loops for patterns (rows, columns, circular arrangements)
   - Calculate positions parametrically based on loop indices
   - Example: `for row in range(n_rows): for i in range(n_holes):`

5. **Define Explicit Parameters**
   - Create clear, purpose-specific parameters (wall_thickness, n_holes, etc.)
   - Avoid complex derived parameters when possible
   - Use descriptive variable names that indicate purpose

6. **Iterative Modification Approach**
   - Build the base geometry first (hollow cylinder)
   - Then iteratively apply modifications (holes) in a controlled loop
   - Check results frequently during development
