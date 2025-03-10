# CadQuery Code Generation Guidelines

## Core Principles for 3D CAD Generation

1. **Precision in Geometry Positioning**
   - Use mathematical approaches (trigonometry, parametric equations) for positioning on curved surfaces
   - Convert between coordinate systems thoughtfully (Cartesian, cylindrical, etc.)
   - Ensure continuity and symmetry in patterns on curved surfaces

2. **Effective Boolean Operations**
   - Build complete base geometry before applying complex operations
   - Ensure cutting solids extend fully through target surfaces
   - Verify that all operations result in valid, manifold geometry

3. **Proper Workplane Management**
   - Position workplanes intentionally in 3D space to align with desired features
   - Use transformation matrices to maintain correct orientation on curved surfaces
   - Create local coordinate systems that follow the contours of complex surfaces

## Specific Techniques for Common Patterns

### Placing Features on Cylindrical Surfaces

1. **Projection Method (Preferred for Complex Shapes)**
   ```python
   # Create base shape on a flat workplane
   shape_wire = (cq.Workplane("XZ", origin=(0, radius, z_position))
                 .polygon(6, size)
                 .wires()
                 .val())
   
   # Project onto cylindrical surface
   projected_wire = shape_wire.projectToShape(
       targetObject=cylinder.val(), 
       center=(0, 0, z_position)
   )[0]
   
   # Rotate around cylinder axis for multiple instances
   rotated_wire = projected_wire.rotate((0, 0, 0), (0, 0, 1), angle)
   ```

2. **Direct Transformation Method (More Intuitive)**
   ```python
   for i in range(n_features):
       angle_deg = i * 360.0 / n_features
       angle_rad = math.radians(angle_deg)
       
       # Position on cylinder surface
       x = radius * math.cos(angle_rad)
       y = radius * math.sin(angle_rad)
       
       # Create a feature on a transformed workplane
       feature = (cq.Workplane("XY")
                  .transformed(offset=(x, y, z), rotate=(0, 90, angle_deg))
                  .polygon(6, size)
                  .extrude(depth, both=True))
   ```

### Creating Hollow Objects

1. **Shell Method (For Complex Shapes)**
   ```python
   # Create solid and then hollow it out
   solid = cq.Workplane("XY").box(10, 10, 10)
   hollow = solid.shell(-1)  # Negative value hollows inward
   ```

2. **Cut Method (For Simple Shapes)**
   ```python
   # Create outer shape then cut out inner shape
   outer = cq.Workplane("XY").circle(outer_radius).extrude(height)
   inner = cq.Workplane("XY").circle(inner_radius).extrude(height)
   hollow = outer.cut(inner)
   ```

3. **Face Thickening (For Surface-Based Models)**
   ```python
   # Extract a face and thicken it
   surface = cylinder.faces("not %Plane").val()
   shell = surface.thicken(thickness)
   ```

### Distributing Features on Surfaces

1. **Even Distribution Around Circumference**
   ```python
   for i in range(n_features):
       angle = i * 360.0 / n_features
       # Position feature at angle
   ```

2. **Grid Patterns on Curved Surfaces**
   ```python
   for row in range(n_rows):
       z = row * row_spacing
       for col in range(n_columns):
           angle = col * 360.0 / n_columns
           # Offset alternate rows for better distribution
           if row % 2 == 1:
               angle += 360.0 / (2 * n_columns)
   ```

3. **Maintaining Feature Orientation on Curved Surfaces**
   - Always orient workplanes to be normal (perpendicular) to the target surface
   - For cylindrical surfaces, rotate the workplane to match the angle around the cylinder
   - Use the `.transformed()` method with correct rotation parameters

## Common Errors and Solutions

1. **"No Pending Wires" Errors**
   - Ensure wire objects are properly created before operations
   - Use `.wires().val()` to extract wire objects from workplanes
   - Verify that operations on wires return valid wire objects

2. **Boolean Operation Failures**
   - Check that all objects involved in boolean operations are valid solids
   - Ensure cutting objects extend fully through the target object
   - Verify that there's actual intersection between objects

3. **Projection Failures**
   - Ensure the target shape is a valid solid or face
   - Position the source shape close to the target surface
   - Specify a valid center point for projection
   - Validate that projected shapes are returned correctly

4. **Object Orientation Issues**
   - Use the full rotation matrix in `.transformed()` calls
   - Remember rotation order matters (XYZ vs. ZYX)
   - Double-check rotation angles (degrees vs. radians)

## Performance Optimization

1. **Use Face Operations When Possible**
   - Operating on faces is more efficient than on solids
   - Extract specific faces using selectors: `.faces("not %Plane").val()`
   - Apply operations like `.makeHoles()` or `.thicken()` on faces

2. **Batch Similar Operations**
   - Create arrays of features and apply them in batches
   - Use list comprehensions for generating multiple objects
   - Combine multiple operations where possible

## Code Structure Best Practices

1. **Define All Parameters at the Top**
   - Include clear, descriptive variable names
   - Document units of measurement
   - Establish relationships between parameters

2. **Use Functions for Repeated Patterns**
   - Create helper functions for generating specific features
   - Parameterize functions to make them reusable
   - Return valid CadQuery objects from functions

3. **Include Validation Steps**
   - Check `.isValid()` on complex objects
   - Add assertions to verify expected dimensions or properties
   - Include debug visualization during development