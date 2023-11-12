bl_info = {
    "name": "Bakar Tools",
    "blender": (4, 0, 0),
    "category": "Object",
}

import bpy
import bmesh
import random
import mathutils
import numpy as np
import colorsys
import math
import importlib

# =============================================================================
# Converting Triangles to Quads Operator
# =============================================================================
class ConvertQuadsOperator(bpy.types.Operator):
    bl_idname = "object.convert_quads_operator"
    bl_label = "Convert to Regular Quads"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        
        if selected_objects:
            for obj in selected_objects:
                self.convert_triangulated_quads_to_regular_quads(obj)
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Please select one or more mesh objects.")
            return {'CANCELLED'}

    def convert_triangulated_quads_to_regular_quads(self, obj):
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)

        # Join triangles into quads
        bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=math.pi, angle_shape_threshold=math.pi)
        
        # Handle leftover triangles by examining local topology
        self.handle_leftover_triangles(bm)
        
        # Update the mesh
        bm.to_mesh(me)
        me.update()

    def handle_leftover_triangles(self, bm):
        triangle_clusters = []

        # Identify triangle clusters
        for face in bm.faces:
            if len(face.verts) == 3:
                cluster = [face]
                # Collect adjacent triangles for this face
                for edge in face.edges:
                    for linked_face in edge.link_faces:
                        if len(linked_face.verts) == 3 and linked_face not in cluster:
                            cluster.append(linked_face)
                if cluster not in triangle_clusters:
                    triangle_clusters.append(cluster)

        # Sort clusters by size to prioritize larger ones
        triangle_clusters.sort(key=len, reverse=True)

        # Process each triangle cluster
        for cluster in triangle_clusters:
            if len(cluster) == 1:
                # Handle isolated triangles differently, e.g., by checking their neighbors' structure 
                # and deciding the best course of action.
                self.process_isolated_triangle(bm, cluster[0])
            elif len(cluster) > 1:
                # Handle larger clusters. This is where we can apply more sophisticated logic based on
                # the local topology and edge flow.
                self.process_triangle_cluster(bm, cluster)

    def process_isolated_triangle(self, bm, triangle):
        # Placeholder for handling isolated triangles
        pass

    def process_triangle_cluster(self, bm, cluster):
        # The strategy for processing triangle clusters can be complex and might involve
        # examining the surrounding quads, the direction of edge flow, and the overall layout
        # of vertices in the cluster.

        # As a starting point, try to join the triangles in the cluster
        try:
            bmesh.ops.join_triangles(bm, faces=cluster, angle_face_threshold=math.pi, angle_shape_threshold=math.pi)
        except:
            # Here we can add more refined handling, e.g., flipping specific edges, collapsing vertices,
            # or even adding new vertices to facilitate better quad layouts.
            pass


# =============================================================================
# Non-Manifold Geometry Detector Operator
# =============================================================================
class NonManifoldGeometryDetector(bpy.types.Operator):
    bl_idname = "object.non_manifold_geometry_detector"
    bl_label = "Non-Manifold Geometry Detector"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object

        if obj and obj.type == 'MESH':
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Ensure we're in edge selection mode
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
            
            # Select non-manifold edges
            bpy.ops.mesh.select_non_manifold()
            
            # Count selected edges to report
            bpy.ops.object.mode_set(mode='OBJECT')
            count = len([e for e in obj.data.edges if e.select])
            
            # Remain in edit mode to show the highlighted non-manifold edges
            bpy.ops.object.mode_set(mode='EDIT') 
            
            if count == 0:
                self.report({'INFO'}, "No non-manifold geometry found.")
            else:
                self.report({'INFO'}, f"{count} non-manifold edges found.")
            
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}


# =============================================================================
# Topology Pole Checker Operator
# =============================================================================
class TopologyCheckerOperator(bpy.types.Operator):
    bl_idname = "object.topology_checker_operator"
    bl_label = "Topology Checker"
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def find_complex_poles(obj):
        complex_poles = []

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        for v in bm.verts:
            edges = v.link_edges
            if len(edges) > 5:
                complex_poles.append(v.index)

        bm.free()

        return complex_poles

    def execute(self, context):
        obj = context.active_object

        if obj and obj.type == 'MESH':
            complex_poles = self.find_complex_poles(obj)
            if not complex_poles:
                self.report({'INFO'}, "No complex poles found.")
                return {'FINISHED'}
            
            # Switch to edit mode to select vertices
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')

            # Switch back to object mode for data access
            bpy.ops.object.mode_set(mode='OBJECT')
            for v_idx in complex_poles:
                obj.data.vertices[v_idx].select = True

            # Switch back to edit mode to highlight the problem areas
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'INFO'}, f"{len(complex_poles)} complex poles found.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}

# =============================================================================
# Face Orientation Display Operator
# =============================================================================
class FaceOrientationDisplayOperator(bpy.types.Operator):
    bl_idname = "object.face_orientation_display_operator"
    bl_label = "Face Orientation Display"
    bl_description = "Toggle or set the face orientation display in the 3D viewport."
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
        items=[
            ("TOGGLE", "Toggle", "Toggle face orientation display"),
            ("ON", "On", "Turn on face orientation display"),
            ("OFF", "Off", "Turn off face orientation display")
        ],
        default="TOGGLE",
        description="Action to perform"
    )

    @classmethod
    def poll(cls, context):
        # Ensure there's an active object, it's a mesh, and the current space is the 3D view
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and 
                context.area.type == 'VIEW_3D')

    def execute(self, context):
        # Get the current space
        space = context.area.spaces.active

        # Check if it's the 3D view
        if not isinstance(space, bpy.types.SpaceView3D):
            self.report({'ERROR'}, "Not in a 3D view context.")
            return {'CANCELLED'}

        # Determine action
        if self.action == "TOGGLE":
            space.overlay.show_face_orientation = not space.overlay.show_face_orientation
        elif self.action == "ON":
            space.overlay.show_face_orientation = True
        else:
            space.overlay.show_face_orientation = False

        # Report the new state
        state = "ON" if space.overlay.show_face_orientation else "OFF"
        self.report({'INFO'}, f"Face Orientation Display: {state}")

        return {'FINISHED'}


# =============================================================================
# Automatic Turntable Operator
# =============================================================================
class SetupTurntableOperator(bpy.types.Operator):
    bl_idname = "object.setup_turntable_operator"
    bl_label = "Set up Turntable"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.setup_turntable()
        return {'FINISHED'}

    def apply_material_to_object(self, material, obj):
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    def setup_turntable(self):
        import math
        from mathutils import Vector

        if bpy.context.selected_objects == []:
            print("No object selected. Please select a 3D model to proceed.")
            return

        active_object = bpy.context.active_object

        # Camera setup
        bpy.ops.object.camera_add(location=(active_object.location.x + 10, active_object.location.y, active_object.location.z))
        camera = bpy.context.object
        camera.name = "Turntable_Camera"
        
        direction = active_object.location - camera.location
        rot_quat = direction.to_track_quat('Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        camera.rotation_euler.z += math.pi
        bpy.context.scene.camera = camera

        # Add empty and set its name
        bpy.ops.object.empty_add(location=active_object.location)
        empty = bpy.context.object
        empty.name = "Turntable_Empty"

        # Parenting and constraints setup
        camera.parent = empty
        camera.location.z = 0

        track_to = camera.constraints.new(type='TRACK_TO')
        track_to.target = active_object
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_Y'
        
        # Add shader balls and parent to camera
        for loc, is_metallic in [((0.18, -0.13, -1), False), ((0.3, -0.13, -1), True)]:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 0))
            shader_ball = bpy.context.object
            shader_ball.parent = camera
            shader_ball.location = loc
            shader_ball.scale = (0.1, 0.1, 0.1)
            
            # Smooth shading and subdivision modifier
            bpy.ops.object.shade_smooth()
            bpy.ops.object.modifier_add(type='SUBSURF')
            shader_ball.modifiers["Subdivision"].levels = 2
            shader_ball.modifiers["Subdivision"].render_levels = 2
            
            # Create and assign material
            mat = bpy.data.materials.new(name="ShaderBallMaterial")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            shader = nodes.get('Principled BSDF')
            if is_metallic:
                shader.inputs['Metallic'].default_value = 1.0
                shader.inputs['Roughness'].default_value = 0.0
            self.apply_material_to_object(mat, shader_ball)

        # Animation setup
        start_frame = 0
        end_frame = 250

        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = end_frame

        empty.rotation_euler.z = -math.pi / 2
        empty.keyframe_insert(data_path="rotation_euler", frame=start_frame)

        empty.rotation_euler.z = 2 * math.pi
        empty.keyframe_insert(data_path="rotation_euler", frame=end_frame)

        empty.rotation_euler.z = 3 * math.pi / 2
        empty.keyframe_insert(data_path="rotation_euler", frame=end_frame)

        for fcurve in empty.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'LINEAR'

        # Cleanup and final constraints
        if "Track To" in camera.constraints:
            camera.constraints.remove(camera.constraints["Track To"])

        track_to = camera.constraints.new(type='TRACK_TO')
        track_to.target = empty
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_Y'

        camera.data.lens = 50

        print("Turntable setup complete with shader balls.")


# =============================================================================
# Mesh Quality Info Operator
# =============================================================================
import bpy
import bmesh
import numpy as np

class MeshQualityDialogOperator(bpy.types.Operator):
    bl_idname = "object.mesh_quality_dialog_operator"
    bl_label = "Select faces with aspect ratio > 2?"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

class MeshQualityInfoOperator(bpy.types.Operator):
    bl_idname = "object.mesh_quality_info_operator"
    bl_label = "Mesh Quality Info"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object

        if obj.type != 'MESH':
            self.report({'ERROR'}, "Selected object is not a mesh.")
            return {'CANCELLED'}

        aspect_ratios = []
        quad_count = 0
        non_quad_count = 0
        problematic_faces = 0
        threshold_ratio = 2.01

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        for face in bm.faces:
            face.select_set(False)
            if len(face.verts) == 4:
                quad_count += 1
                lengths = [edge.calc_length() for edge in face.edges]
                min_len = min(lengths)
                max_len = max(lengths)
                aspect_ratio = max_len / min_len
                aspect_ratios.append(aspect_ratio)

                if aspect_ratio > threshold_ratio:
                    problematic_faces += 1
                    face.select = True

            else:
                non_quad_count += 1

        bm.select_flush(True)
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()

        total_faces = quad_count + non_quad_count
        if total_faces == 0:
            self.report({'ERROR'}, "No faces found.")
            return {'CANCELLED'}

        worst_aspect_ratio = max(aspect_ratios) if aspect_ratios else 0
        quad_percentage = (quad_count / total_faces) * 100

        quality = "N/A"
        if worst_aspect_ratio <= 1.1:
            quality = "Excellent"
        elif problematic_faces == 0:
            quality = "Good"
        else:
            quality = "Needs Attention"

        info_text = f"""Total Faces: {total_faces}
        Quads: {quad_count} ({quad_percentage:.2f}%)
        Non-Quads: {non_quad_count}
        Worst Aspect Ratio: {worst_aspect_ratio:.2f}
        Problematic Faces: {problematic_faces}
        Mesh Quality: {quality}"""

        self.report({'INFO'}, info_text)

        if problematic_faces > 0:
            bpy.ops.object.mesh_quality_dialog_operator('INVOKE_DEFAULT')
        
        return {'FINISHED'}

# =============================================================================
# Polycount Operator
# =============================================================================
class PolycountOperator(bpy.types.Operator):
    bl_idname = "object.polycount_operator"
    bl_label = "Polycount (Tris)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected_objects = context.selected_objects

        # Filter out only mesh type objects
        mesh_objects = [obj for obj in selected_objects if obj.type == 'MESH']

        if not mesh_objects:
            self.report({'ERROR'}, "No mesh objects selected.")
            return {'CANCELLED'}

        total_tris = 0

        for obj in mesh_objects:
            # Check for subdivision modifiers and disable them temporarily
            subdivisions = [mod for mod in obj.modifiers if mod.type == 'SUBSURF']
            original_values = {mod: mod.show_viewport for mod in subdivisions}
            for mod in subdivisions:
                mod.show_viewport = False

            # Update the mesh to reflect the changes
            obj.data.update()

            # Calculate the number of triangles for the current object
            tris_count = sum(1 for poly in obj.data.polygons if len(poly.vertices) == 3)
            quads_count = sum(1 for poly in obj.data.polygons if len(poly.vertices) == 4)
            tris_from_quads = quads_count * 2
            total_tris += tris_count + tris_from_quads

            # Restore the original state of subdivision modifiers
            for mod, value in original_values.items():
                mod.show_viewport = value

        # Report the total triangles count for all selected meshes
        self.report({'INFO'}, f"Total Polycount (Tris) in selected meshes: {total_tris}")

        return {'FINISHED'}

# =============================================================================
# Vertex Paint Operator
# =============================================================================
class VertexPaintOperator(bpy.types.Operator):
    bl_idname = "object.vertex_paint_operator"
    bl_label = "Vertex Paint Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        initial_mode = obj.mode

        if obj and obj.type == 'MESH':
            bpy.ops.object.mode_set(mode='VERTEX_PAINT')

            # Get the active color from the color picker
            color = bpy.data.brushes["Draw"].color

            # Append alpha value
            color = color[0], color[1], color[2], 1

            # Create a new color layer if it doesn't exist
            if not obj.data.vertex_colors:
                obj.data.vertex_colors.new()

            color_layer = obj.data.vertex_colors.active

            # Paint selected polygons with the active color
            for poly in obj.data.polygons:
                if poly.select:
                    for loop_index in poly.loop_indices:
                        color_layer.data[loop_index].color = color

            bpy.ops.object.mode_set(mode=initial_mode)
            self.report({'INFO'}, "Vertex painted selected polygons with selected vertex paint color.")
            return {'FINISHED'}

        else:
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}


# =============================================================================
# Show Ngons Operator
# =============================================================================
class ShowNgonsOperator(bpy.types.Operator):
    bl_idname = "object.show_ngons_operator"
    bl_label = "Show Ngons"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        initial_mode = obj.mode
        found = 0

        if obj and obj.type == 'MESH':
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            
            for polygon in obj.data.polygons:
                if len(polygon.vertices) > 4:
                    polygon.select = True
                    found += 1
            
            bpy.ops.object.mode_set(mode='EDIT')
            
            if found == 0:
                self.report({'INFO'}, "No ngons found.")
                if initial_mode == 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
            else:
                self.report({'INFO'}, f"{found} ngons found.")
            
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}

# =============================================================================
# Show Tris Operator
# =============================================================================
class ShowTrisOperator(bpy.types.Operator):
    bl_idname = "object.show_tris_operator"
    bl_label = "Show Tris"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        initial_mode = obj.mode
        found = 0

        if obj and obj.type == 'MESH':
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            for polygon in obj.data.polygons:
                if len(polygon.vertices) == 3:
                    polygon.select = True
                    found += 1
            
            bpy.ops.object.mode_set(mode='EDIT')
            
            if found == 0:
                self.report({'INFO'}, "No tris found.")
                if initial_mode == 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
            else:
                self.report({'INFO'}, f"{found} tris found.")
            
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}


# =============================================================================
# Edge Flow Operator
# =============================================================================
class EdgeFlowToolsOperator(bpy.types.Operator):
    bl_idname = "object.edge_flow_tools_operator"
    bl_label = "Set Edge Flow"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        # Ensure context is correct
        if context.mode != 'EDIT_MESH':
            self.report({'WARNING'}, "Must be in Edit Mode.")
            return {'CANCELLED'}
        
        # Set edge flow
        bpy.ops.mesh.set_edge_flow()

        return {'FINISHED'}


# =============================================================================
# Bridge Operator
# =============================================================================
class BridgeOperator(bpy.types.Operator):
    """Bridges between two edge loops or arbitrary selected regions"""
    bl_idname = "object.bridge_operator"
    bl_label = "Bridge Edges"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh.")
            return {'CANCELLED'}

        # Switch to edit mode
        initial_mode = obj.mode
        bpy.ops.object.mode_set(mode='EDIT')

        # Check if there's a selection
        bm = bmesh.from_edit_mesh(obj.data)
        edges = [e for e in bm.edges if e.select]

        if len(edges) < 2:
            self.report({'ERROR'}, "Please select regions or edge loops to bridge.")
            bpy.ops.object.mode_set(mode=initial_mode)
            return {'CANCELLED'}

        # Try to bridge the selected regions
        try:
            bpy.ops.mesh.bridge_edge_loops()
            self.report({'INFO'}, "Bridged selected regions.")
        except:
            self.report({'ERROR'}, "Failed to bridge the selected regions.")
            bpy.ops.object.mode_set(mode=initial_mode)
            return {'CANCELLED'}

        # Switch back to the initial mode
        bpy.ops.object.mode_set(mode=initial_mode)
        return {'FINISHED'}


# =============================================================================
# Cryptomatte Vertex Paint Operator
# =============================================================================
class CryptomatteVertexPaint:
    def __init__(self):
        self.used_colors = set()

    def generate_unique_random_color(self):
        while True:
            hue = random.uniform(0, 1)
            saturation = random.uniform(0.4, 1)
            value = random.uniform(0.4, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            color_str = f"{r:.4f}-{g:.4f}-{b:.4f}"
            if color_str not in self.used_colors:
                self.used_colors.add(color_str)
                return r, g, b

    def apply_vertex_colors(self, mesh):
        if not mesh.vertex_colors:
            mesh.vertex_colors.new()
        
        color_layer = mesh.vertex_colors.active
        visited_faces = set()
        
        vert_to_faces = {}
        for poly in mesh.polygons:
            for vert in poly.vertices:
                if vert not in vert_to_faces:
                    vert_to_faces[vert] = []
                vert_to_faces[vert].append(poly.index)
        
        for poly in mesh.polygons:
            if poly.index not in visited_faces:
                r, g, b = self.generate_unique_random_color()
                connected_faces = self.get_connected_faces(mesh, poly.index, vert_to_faces)
                visited_faces.update(connected_faces)
                
                for face_index in connected_faces:
                    for loop_index in mesh.polygons[face_index].loop_indices:
                        color_layer.data[loop_index].color = (r, g, b, 1.0)

    def get_connected_faces(self, mesh, start_face_index, vert_to_faces):
        stack = [start_face_index]
        visited = set()
        connected_faces = set()

        while stack:
            face_index = stack.pop()
            visited.add(face_index)
            connected_faces.add(face_index)

            for vert_index in mesh.polygons[face_index].vertices:
                for neighbor_face in vert_to_faces[vert_index]:
                    if neighbor_face not in visited:
                        stack.append(neighbor_face)
                        visited.add(neighbor_face)

        return connected_faces

class CryptomatteVertexPaintOperator(bpy.types.Operator):
    bl_idname = "object.cryptomatte_vertex_paint"
    bl_label = "Object Cryptomatte Painting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cryptomatte = CryptomatteVertexPaint()

        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                cryptomatte.apply_vertex_colors(obj.data)

        return {'FINISHED'}


# =============================================================================
# Rename Objects Operator
# =============================================================================
class RenameObjectsOperator(bpy.types.Operator):
    bl_idname = "object.rename_objects_operator"
    bl_label = "Rename Objects for Production"
    bl_options = {'REGISTER', 'UNDO'}

    base_name: bpy.props.StringProperty(
        name="Base Name",
        description="Base name for the objects",
        default="OBJ"
    )

    def execute(self, context):
        selected_objs = context.selected_objects
        if not selected_objs:
            self.report({'ERROR'}, "No objects selected.")
            return {'CANCELLED'}

        for obj in selected_objs:
            new_name = self.get_new_name(obj)
            obj.name = new_name

            # Renaming child objects
            for index, child in enumerate(obj.children):
                if child in selected_objs:
                    child.name = f"{new_name}_Child_{str(index).zfill(2)}"

        self.report({'INFO'}, "Objects renamed successfully.")
        return {'FINISHED'}

    def get_new_name(self, obj):
        type_prefix = self.get_type_prefix(obj.type)
        return f"{type_prefix}{self.base_name}"

    def get_type_prefix(self, obj_type):
        prefixes = {
            'MESH': 'MESH_',
            'CAMERA': 'CAM_',
            'LIGHT': 'LGT_',
            # ... Add other types as needed ...
        }
        return prefixes.get(obj_type, "")

###########################################################################################################################
##################################################################################################################
##################################################################################################################
###########################################################################################################################

# ------------------------------------
#       PREFERENCES CLASS
# ------------------------------------
class MeshMintToolsPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    # Add any other addon-specific preferences here

    def draw(self, context):
        layout = self.layout
        layout.label(text="Bakar Tools Preferences")
        layout.operator("object.bakar_tools_update")
        
class MeshMintToolsUpdateOperator(bpy.types.Operator):
    bl_idname = "object.bakar_tools_update"
    bl_label = "Update Bakar Tools"

    def execute(self, context):
        # Logic to refresh or reload the script
        reload_addon_modules()
        self.report({'INFO'}, "Addon updated successfully.")
        return {'FINISHED'}
    
def reload_addon_modules():
    import sys

    for module_name, module in sys.modules.items():
        if module_name.startswith("your_addon_package_name"):
            importlib.reload(module)

    print("Addon modules reloaded.")


# =============================================================================
# The MeshMintTools Panel in N-Panel Operator
# =============================================================================
class MeshMintToolsPanel(bpy.types.Panel):
    bl_label = "MeshMint"
    bl_idname = "OBJECT_PT_meshmint"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MeshMint'

    def draw(self, context):
        layout = self.layout

        # Utility Subsection
        utility_box = layout.box()
        utility_box.label(text="Utility")
        
        utility_box.operator("object.setup_turntable_operator", text="Set up Turntable")
        utility_box.operator("object.mesh_quality_info_operator", text="Mesh Quality Info")
        utility_box.operator("object.polycount_operator", text="Get Polycount")

        # Mesh Analysis Subsection
        mesh_analysis_box = layout.box()
        mesh_analysis_box.label(text="Mesh Analysis")
        row = mesh_analysis_box.row(align=True)
        row.operator("object.show_ngons_operator", text="Show Ngons")
        row.operator("object.show_tris_operator", text="Show Tris")
        mesh_analysis_box.operator("object.non_manifold_geometry_detector", text="Detect Non-Manifold Geo")
        mesh_analysis_box.operator("object.topology_checker_operator", text="Detect Poles")
        mesh_analysis_box.operator("object.face_orientation_display_operator", text="Toggle Face Orientation")

        # Vertex Painting Subsection
        vertex_painting_box = layout.box()
        vertex_painting_box.label(text="Vertex Painting")
        vertex_painting_box.operator("object.vertex_paint_operator", text="Vertex Paint Selected")
        vertex_painting_box.operator("object.cryptomatte_vertex_paint", text="Cryptomatte Vertex Paint")

        # Modeling Utility Subsection
        box = layout.box()
        box.label(text="Modeling Tools")
        box.operator("object.edge_flow_tools_operator", text="Set Edge Flow")
        box.operator("object.convert_quads_operator", text="Convt Sel Obj(s) to Quads")
        box.operator("object.bridge_operator", text="Bridge Edges")
        
        # Rename Subsection
        rename_box = layout.box()
        rename_box.label(text="Renaming Tools")
        rename_box.prop(context.window_manager, "base_name")
        rename_box.operator("object.rename_objects_operator", text="Rename for Production")

# =============================================================================
# Register/Unregister Classes
# =============================================================================
def register():
    bpy.utils.register_class(ConvertQuadsOperator)
    bpy.utils.register_class(MeshMintToolsPanel)
    bpy.utils.register_class(SetupTurntableOperator)
    bpy.utils.register_class(MeshQualityInfoOperator)
    bpy.utils.register_class(ShowNgonsOperator)
    bpy.utils.register_class(ShowTrisOperator)
    bpy.utils.register_class(VertexPaintOperator)
    bpy.utils.register_class(PolycountOperator)
    bpy.utils.register_class(NonManifoldGeometryDetector)
    bpy.utils.register_class(TopologyCheckerOperator)
    bpy.utils.register_class(FaceOrientationDisplayOperator)
    bpy.utils.register_class(EdgeFlowToolsOperator)
    bpy.utils.register_class(BridgeOperator)
    bpy.utils.register_class(CryptomatteVertexPaintOperator)
    bpy.utils.register_class(MeshQualityDialogOperator)
    bpy.utils.register_class(RenameObjectsOperator)
    bpy.utils.register_class(MeshMintToolsPreferences)
    bpy.utils.register_class(MeshMintToolsUpdateOperator)

def unregister():
    bpy.utils.unregister_class(ConvertQuadsOperator)
    bpy.utils.unregister_class(MeshMintToolsPanel)
    bpy.utils.unregister_class(SetupTurntableOperator)
    bpy.utils.unregister_class(MeshQualityInfoOperator)
    bpy.utils.unregister_class(ShowNgonsOperator)
    bpy.utils.unregister_class(ShowTrisOperator)
    bpy.utils.unregister_class(VertexPaintOperator)
    bpy.utils.unregister_class(PolycountOperator)
    bpy.utils.unregister_class(NonManifoldGeometryDetector)
    bpy.utils.unregister_class(TopologyCheckerOperator)
    bpy.utils.unregister_class(FaceOrientationDisplayOperator)
    bpy.utils.unregister_class(EdgeFlowToolsOperator)
    bpy.utils.unregister_class(BridgeOperator)
    bpy.utils.unregister_class(CryptomatteVertexPaintOperator)
    bpy.utils.unregister_class(MeshQualityDialogOperator)
    bpy.utils.unregister_class(RenameObjectsOperator)
    bpy.utils.unregister_class(MeshMintToolsPreferences)
    bpy.utils.unregister_class(MeshMintToolsUpdateOperator)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        unregister()  # Just in case
    except Exception as e:
        print(f"Could not unregister: {e}")
    register()