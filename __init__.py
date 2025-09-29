bl_info = {
	"name": "Archean (Galaxy4D)",
	"author": "Xenon3D inc.",
	"version": (1, 0),
	"blender": (2, 80, 0),
	"location": "View3D > Sidebar > Archean (Galaxy4D) tab",
	"description": "Archean's Blender plugin",
	"warning": "",
	"wiki_url": "",
	"category": "Object"
}

import bpy
import os
import configparser
import math
import struct
from glob import glob
from .adapterMesh import *
from mathutils import Vector
from mathutils import Matrix

def recrop_thumbnail_image(file_path, maxWidth=128, maxHeight=128, forceExactDimensions=True):
	image = bpy.data.images.load(file_path)

	width = image.size[0]
	height = image.size[1]

	# Create a new image buffer and copy pixel data
	image_pixels = list(image.pixels[:])

	min_x = width
	min_y = height
	max_x = 0
	max_y = 0

	# Find the bounding box of non-transparent pixels
	for y in range(height):
		for x in range(width):
			# Get the alpha value of the pixel
			index = (y * width + x) * 4 + 3
			alpha = image_pixels[index]

			# If the pixel is not transparent
			if alpha > 0:
				min_x = min(min_x, x)
				max_x = max(max_x, x)
				min_y = min(min_y, y)
				max_y = max(max_y, y)

	# Crop the image based on the bounding box
	cropped_width = max_x - min_x + 1
	cropped_height = max_y - min_y + 1
	cropped_pixels = [0] * 4 * cropped_width * cropped_height

	for y in range(cropped_height):
		for x in range(cropped_width):
			index = ((min_y + y) * width + (min_x + x)) * 4
			new_index = (y * cropped_width + x) * 4
			cropped_pixels[new_index:new_index+4] = image_pixels[index:index+4]

	# Replace the original image data with the cropped data
	image.scale(cropped_width, cropped_height)
	image.pixels = cropped_pixels

	# Calculate the resize dimensions while keeping aspect ratio
	aspect_ratio = cropped_width / cropped_height
	if cropped_width > cropped_height:
		new_width = maxWidth
		new_height = math.ceil(maxHeight / aspect_ratio)
	else:
		new_height = maxHeight
		new_width = math.ceil(maxWidth * aspect_ratio)

	# Resize the image
	image.scale(new_width, new_height)

	if forceExactDimensions:
		# Create a new image with exactly the desired dimensions
		final_image = bpy.data.images.new("final", maxWidth, maxHeight, alpha=True)
		final_image.pixels = [0.0, 0.0, 0.0, 0.0] * maxWidth * maxHeight

		# Calculate margins for centering
		margin_x = (maxWidth - new_width) // 2
		margin_y = (maxHeight - new_height) // 2

		# Copy the resized image into the center of the new image
		for y in range(new_height):
			for x in range(new_width):
				index = (y * new_width + x) * 4
				new_index = ((margin_y + y) * maxWidth + (margin_x + x)) * 4
				final_image.pixels[new_index:new_index+4] = image.pixels[index:index+4]

		# Save the final image back to the original file path
		final_image.filepath_raw = file_path
		final_image.file_format = 'PNG'
		final_image.save()
	else:
		# Save the final image back to the original file path
		image.save()

	# Unload the image (optional)
	bpy.data.images.remove(image)

def update_legacy_materials():
	for mat in bpy.data.materials:
		if mat.use_nodes:
			nodes = mat.node_tree.nodes
			principled = nodes.get('Principled BSDF')
			if principled is not None:
				if 'Base Color' in principled.inputs and hasattr(principled.inputs['Base Color'], "default_value"):
					mat.diffuse_color = principled.inputs['Base Color'].default_value
				if 'Metallic' in principled.inputs and hasattr(principled.inputs['Metallic'], "default_value"):
					mat.metallic = principled.inputs['Metallic'].default_value
				if 'Roughness' in principled.inputs and hasattr(principled.inputs['Roughness'], "default_value"):
					mat.roughness = principled.inputs['Roughness'].default_value
			else:
				print("Material '" + mat.name + "' does not have a Principled BSDF node")


def render_and_save_thumbnail(obj, show_outlines = False):
	update_legacy_materials()
	
	# Get the path of the current blend file
	blend_file_path = bpy.data.filepath
	directory, blend_file = os.path.split(blend_file_path)
	
	bbox = get_recursive_renderable_bounds(obj)
	assert bbox is not None, "No renderable object found in entity '" + obj.name + "'"
	
	bbox_center = (bbox['min'] + bbox['max']) / 2
	bbox_size = bbox['max'] - bbox['min']
	bbox_size = Vector((max(bbox_size.x, 0.125), max(bbox_size.y, 0.125), max(bbox_size.z, 0.125)))
	assert bbox_size.length > 0, "Object '" + obj.name + "' has zero size"

	# Create a temporary scene
	temp_scene = bpy.data.scenes.new("TempScene")
	
	# Recursively link the object and all its descendants to the temporary scene
	def link_with_descendants(obj, scene):
		if getattr(obj, "G4D_IS_RENDERABLE", False):
			scene.collection.objects.link(obj)
		for child in obj.children:
			child_bbox = link_with_descendants(child, scene)
	link_with_descendants(obj, temp_scene)
	
	# Create a camera
	cam_data = bpy.data.cameras.new('TempCamera')
	cam = bpy.data.objects.new('TempCamera', cam_data)
	temp_scene.collection.objects.link(cam)

	# Set camera lens angle
	cam_data.angle = math.radians(90.0)
	
	# Set camera position based on the bounding box size
	cam.location = bbox_center + bbox_size.length * 1.5 * Vector((1, 1.5, 1.5)).normalized()

	# Point camera to the bounding box center
	direction = (bbox_center - cam.location).normalized() # Using the bounding box center
	rot_quat = direction.to_track_quat('-Z', 'Y')
	cam.rotation_euler = rot_quat.to_euler()

	# Set this camera as the scene's active camera
	temp_scene.camera = cam

	# Enable Freestyle
	temp_scene.render.use_freestyle = False

	# Set the render engine
	temp_scene.render.engine = 'BLENDER_WORKBENCH' # BLENDER_EEVEE_NEXT
	temp_scene.render.use_compositing = False
	temp_scene.display.shading.light = 'STUDIO'
	temp_scene.display.shading.color_type = 'TEXTURE'
	temp_scene.display.shading.type = 'SOLID'
	temp_scene.display.shading.background_color = (0,0,0)
	temp_scene.display.shading.show_specular_highlight = True

	# Create a light
	light_data = bpy.data.lights.new('TempLight', 'SUN')
	light = bpy.data.objects.new('TempLight', light_data)
	temp_scene.collection.objects.link(light)
	light.location = cam.location
	light.rotation_euler = cam.rotation_euler
	light.data.energy = 10
	light.data.use_shadow = False
	light.data.color = (1, 1, 1)

	if show_outlines:
		linestyle = bpy.data.linestyles.new('Outline')
		linestyle.color = (0, 0.3, 0.3)
		linestyle.thickness = 1.0
		lineset = temp_scene.view_layers[0].freestyle_settings.linesets.new('ObjectOutline')
		lineset.linestyle = linestyle

	# Set resolution
	temp_scene.render.resolution_x = 512
	temp_scene.render.resolution_y = 512
	
	# Set the background to transparent
	temp_scene.render.film_transparent = True
	
	# Set render settings
	temp_scene.render.image_settings.file_format = 'PNG'

	# Set the output path
	thumbnail_path = os.path.join(directory, f"{obj.name}.png")
	temp_scene.render.filepath = thumbnail_path

	# Set the current scene to the temporary one
	original_scene = bpy.context.window.scene
	bpy.context.window.scene = temp_scene

	# Render image (render the object, not animation)
	bpy.ops.render.render(write_still = True)

	# Delete the temporary scene
	bpy.data.scenes.remove(temp_scene)
	bpy.context.window.scene = original_scene
	
	recrop_thumbnail_image(thumbnail_path)

	return thumbnail_path

def export_ini(obj, file_path):
	config = configparser.ConfigParser()
	export_object_and_children_ini(config, obj, file_path)
	with open(file_path+'.ini', 'w') as configfile:
		config.write(configfile)

def cleanup_unused_renderable_export_files(entity_root):
	"""Remove renderable export artifacts left on disk for the provided entity."""
	if entity_root is None:
		return

	entity_name = getattr(entity_root, "name", None)
	if not entity_name:
		return

	export_root = os.path.abspath(bpy.path.abspath("//"))
	entity_prefix = os.path.join(export_root, entity_name)

	for path in glob(f"{entity_prefix}.*.bin"):
		if not os.path.isfile(path):
			continue
		try:
			os.remove(path)
		except OSError as err:
			print(f"Failed to remove unused export file '{path}': {err}")

def prepare_objects_for_export(obj):
	if getattr(obj, "G4D_IS_JOINT", False):
		reset_obj_joint_preview(obj)
	obj.select_set(getattr(obj, "G4D_IS_RENDERABLE", False))
	for child in obj.children:
		prepare_objects_for_export(child)

def entity_has_smooth_shading(entity_obj):
	visited_objects = set()
	visited_meshes = set()

	def check_object(obj):
		if obj is None:
			return False
		obj_ptr = obj.as_pointer()
		if obj_ptr in visited_objects:
			return False
		visited_objects.add(obj_ptr)

		if getattr(obj, "G4D_IS_RENDERABLE", False):
			if obj.type == 'MESH' and obj.data is not None:
				mesh = obj.data
				mesh_ptr = mesh.as_pointer()
				if mesh_ptr not in visited_meshes:
					visited_meshes.add(mesh_ptr)
					if getattr(mesh, "use_auto_smooth", False) or getattr(mesh, "has_custom_normals", False):
						return True
					for poly in mesh.polygons:
						if poly.use_smooth:
							return True
			if obj.instance_type == 'COLLECTION' and obj.instance_collection is not None:
				for inst_obj in obj.instance_collection.objects:
					if check_object(inst_obj):
						return True

		for child in obj.children:
			if check_object(child):
				return True
		return False
	return check_object(entity_obj)


def export_gltf(entityObj, file_path):
	export_normals = entity_has_smooth_shading(entityObj)
	options = {
		'check_existing': True,
		'export_format': 'GLTF_SEPARATE',
		'ui_tab': 'GENERAL',
		'export_copyright': '',
		'export_image_format': 'AUTO',
		'export_texture_dir': '',
		'export_keep_originals': False,
		'export_texcoords': getattr(entityObj, "G4D_ENTITY_GLTF_EXPORT_TEXCOORDS", False),
		'export_normals': export_normals,
		'export_draco_mesh_compression_enable': False,
		'export_draco_mesh_compression_level': 6,
		'export_draco_position_quantization': 14,
		'export_draco_normal_quantization': 10,
		'export_draco_texcoord_quantization': 12,
		'export_draco_color_quantization': 10,
		'export_draco_generic_quantization': 12,
		'export_tangents': False,
		'export_materials': 'EXPORT',
		'export_original_specular': False,
		'export_attributes': False,
		'use_mesh_edges': False,
		'use_mesh_vertices': False,
		'export_cameras': True,
		'use_selection': True,
		'use_visible': False,
		'use_renderable': False,
		'use_active_collection_with_nested': False,
		'use_active_collection': False,
		'use_active_scene': False,
		'export_extras': False,
		'export_yup': True,
		'export_apply': True,
		'export_animations': False,
		'export_frame_range': False,
		'export_frame_step': 1,
		'export_force_sampling': False,
		'export_def_bones': False,
		'export_optimize_animation_size': False,
		'export_anim_single_armature': False,
		'export_reset_pose_bones': False,
		'export_current_frame': False,
		'export_skins': False,
		'export_all_influences': False,
		'export_morph': False,
		'export_morph_normal': False,
		'export_morph_tangent': False,
		'export_lights': False,
		'will_save_settings': False
	}
	bpy.ops.export_scene.gltf(filepath=file_path+'.gltf', **options)

def get_entity_obj(obj):
	if obj is None:
		return None
	if getattr(obj, "G4D_IS_ENTITY", False):
		return obj
	return get_entity_obj(obj.parent)

def get_parent_obj(obj):
	if obj is None:
		return None
	if obj.parent is None:
		return None
	if getattr(obj.parent, "G4D_IS_ENTITY", False):
		return obj.parent
	if getattr(obj.parent, "G4D_IS_JOINT", False):
		return obj.parent
	if getattr(obj.parent, "G4D_IS_RENDERABLE", False):
		return obj.parent
	if getattr(obj.parent, "G4D_IS_COLLIDER", False):
		return obj.parent
	if getattr(obj.parent, "G4D_IS_ADAPTER", False):
		return obj.parent
	if getattr(obj.parent, "G4D_IS_TARGET", False):
		return obj.parent
	if getattr(obj.parent, "G4D_IS_CAMERA", False):
		return obj.parent
	return get_parent_obj(obj.parent)

def get_bounding_box(obj):
	if obj is None:
		return None
	if obj.data is None:
		return None
	if len(obj.data.polygons) == 0:
		return None
	if len(obj.data.vertices) == 0:
		return None
	bb_min = Vector()
	bb_max = Vector()
	for p in obj.data.polygons:
		if len(p.vertices) >= 3:
			for i in p.vertices:
				v = obj.data.vertices[i].co
				bb_min.x = min(bb_min.x, v.x)
				bb_min.y = min(bb_min.y, v.y)
				bb_min.z = min(bb_min.z, v.z)
				bb_max.x = max(bb_max.x, v.x)
				bb_max.y = max(bb_max.y, v.y)
				bb_max.z = max(bb_max.z, v.z)
	return { "min": bb_min, "max": bb_max }

# Recursively compute the bounding box of the object and all its descendants
def get_recursive_renderable_bounds(obj):
	bbox = None
	if getattr(obj, "G4D_IS_JOINT", False):
		reset_obj_joint_preview(obj)
	if getattr(obj, "G4D_IS_RENDERABLE", False):
		bbox = get_bounding_box(obj)
		assert bbox is not None, "Renderable '" + obj.name + "' has no bounds or vertices"
		if obj.instance_type == 'COLLECTION':
			linked_obj = obj.instance_collection.objects[obj.instance_index]
			matrix_world_with_pose = linked_obj.matrix_world.copy()
		else:
			matrix_world_with_pose = obj.matrix_world.copy()
		if obj.pose is not None:
			obj.pose.bones.update()
			matrix_world_with_pose @= obj.pose.matrix
		bbox['min'] = matrix_world_with_pose @ bbox['min']
		bbox['max'] = matrix_world_with_pose @ bbox['max']
	for child in obj.children:
		child_bbox = get_recursive_renderable_bounds(child)
		if bbox is None:
			bbox = child_bbox
		elif child_bbox is not None:
			bbox['min'].x = min(bbox['min'].x, child_bbox['min'].x)
			bbox['min'].y = min(bbox['min'].y, child_bbox['min'].y)
			bbox['min'].z = min(bbox['min'].z, child_bbox['min'].z)
			bbox['max'].x = max(bbox['max'].x, child_bbox['max'].x)
			bbox['max'].y = max(bbox['max'].y, child_bbox['max'].y)
			bbox['max'].z = max(bbox['max'].z, child_bbox['max'].z)
	return bbox
	
def get_recursive_collider_bounds(obj):
	bbox = None
	if getattr(obj, "G4D_IS_JOINT", False):
		reset_obj_joint_preview(obj)
	if getattr(obj, "G4D_IS_COLLIDER", False):
		if obj.instance_type == 'COLLECTION':
			linked_obj = obj.instance_collection.objects[obj.instance_index]
			matrix_world_with_pose = linked_obj.matrix_world.copy()
		else:
			matrix_world_with_pose = obj.matrix_world.copy()
		if obj.pose is not None:
			obj.pose.bones.update()
			matrix_world_with_pose @= obj.pose.matrix
		bbox = get_bounding_box(obj)
		if bbox is not None:
			bbox['min'] = matrix_world_with_pose @ bbox['min']
			bbox['max'] = matrix_world_with_pose @ bbox['max']
	for child in obj.children:
		child_bbox = get_recursive_collider_bounds(child)
		if bbox is None:
			bbox = child_bbox
		elif child_bbox is not None:
			bbox['min'].x = min(bbox['min'].x, child_bbox['min'].x)
			bbox['min'].y = min(bbox['min'].y, child_bbox['min'].y)
			bbox['min'].z = min(bbox['min'].z, child_bbox['min'].z)
			bbox['max'].x = max(bbox['max'].x, child_bbox['max'].x)
			bbox['max'].y = max(bbox['max'].y, child_bbox['max'].y)
			bbox['max'].z = max(bbox['max'].z, child_bbox['max'].z)
	return bbox

def get_obj_position(obj):
	return obj.matrix_local.translation

def get_obj_rotation(obj):
	return obj.matrix_local.to_euler("XYZ")

def add_section_to_config(config, section):
	if section not in config:
		config[section] = {}
def add_property_to_config(config, section, prop, value):
	add_section_to_config(config, section)
	if value is not None:
		if isinstance(value, bpy.types.bpy_prop_array) or isinstance(value, Vector) or isinstance(value, list):
			config.set(section, prop, ' '.join(map(lambda x: "{}".format(x) if isinstance(x, int) else "{:.3f}".format(round(x, 3)), value)))
		else:
			config.set(section, prop, str(value))
def add_parent_to_config(config, section, parentObj):
	if parentObj is not None:
		add_property_to_config(config, section, "parent", parentObj.name)
def add_pos_and_rot_to_config(config, section, obj):
	position = get_obj_position(obj)
	rotation = get_obj_rotation(obj)
	add_property_to_config(config, section, "position", [position.x, position.y, position.z])
	add_property_to_config(config, section, "rotation", [math.degrees(rotation.x), math.degrees(rotation.y), math.degrees(rotation.z)])
def add_pos_and_rot_to_panel(box, obj):
	position = get_obj_position(obj)
	rotation = get_obj_rotation(obj)
	box.row().label(text="Effective position: " + f"{position.x:.3f}, {position.y:.3f}, {position.z:.3f}")
	box.row().label(text="Effective rotation: " + f"{math.degrees(rotation.x):.1f}, {math.degrees(rotation.y):.1f}, {math.degrees(rotation.z):.1f}")

def should_apply_scale(obj):
	if len(obj.children) == 0:
		if obj.data is None:
			return False
		if isinstance(obj.data, bpy.types.Camera):
			return False
	return not (math.isclose(obj.scale.x, 1, rel_tol=1e-6) and
				math.isclose(obj.scale.y, 1, rel_tol=1e-6) and
				math.isclose(obj.scale.z, 1, rel_tol=1e-6))

def should_fix_object_or_children(obj):
	if obj is None:
		return False
	if should_apply_scale(obj):
		return True
	for child in obj.children:
		if should_fix_object_or_children(child):
			return True
	return False

def apply_scale_to_obj_and_children(obj):
	if obj is None:
		return
	hidden = obj.hide_get()
	obj.hide_set(False)
	if should_apply_scale(obj):
		bpy.ops.object.select_all(action='DESELECT')
		obj.select_set(True)
		bpy.context.view_layer.objects.active = obj
		bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
	for child in obj.children:
		apply_scale_to_obj_and_children(child)
	obj.hide_set(hidden)

def get_current_object(context):
	obj = context.active_object
	if obj is None:
		obj = context.view_layer.objects.active
	return obj

class Archean_AddNewEntity(bpy.types.Operator):
	bl_idname = "object.add_archean_entity"
	bl_label = "Add Archean Entity"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		bpy.ops.object.empty_add(type='PLAIN_AXES')
		entity = context.active_object
		entity.name = "New Entity"
		entity["G4D_IS_ENTITY"] = True

		bpy.ops.mesh.primitive_cube_add()
		cube = context.active_object
		cube.parent = entity
		cube["G4D_IS_RENDERABLE"] = True

		return {'FINISHED'}

def add_archean_entity_button(self, context):
	self.layout.operator("object.add_archean_entity", text="Archean Entity", icon='MESH_CUBE')

class Archean_WriteMeshAsPython(bpy.types.Operator):
	bl_idname = "object.write_mesh_as_python"
	bl_label = "Write Mesh as Python"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		obj = context.object
		mesh = obj.data
		dir_path = os.path.dirname(bpy.data.filepath)
		path = os.path.join(dir_path, f"{obj.name}.py")
		
		# Open the output file
		with open(path, 'w') as f:
			# Write some basic imports
			f.write("import bpy\n")
			f.write("from mathutils import Vector\n\n")

			# Write a function to create the mesh
			f.write("def create_mesh(name, parent):\n")

			# Write the vertices
			f.write("\tvertices = [")
			for vertex in mesh.vertices:
				f.write(f"Vector(({vertex.co.x},{vertex.co.y},{vertex.co.z})),")
			f.write("]\n")

			# Write the edges
			f.write("\tedges = [")
			for edge in mesh.edges:
				f.write(f"({edge.vertices[0]},{edge.vertices[1]}),")
			f.write("]\n")

			# Write the polygons
			f.write("\tpolygons = [")
			for polygon in mesh.polygons:
				f.write(f"({','.join(str(v) for v in polygon.vertices)}),")
			f.write("]\n")

			# Write the code to create the mesh object
			f.write("\tmesh_data = bpy.data.meshes.new(name + '_data')\n")
			f.write("\tmesh_data.from_pydata(vertices, edges, polygons)\n")
			f.write("\tmesh_obj = bpy.data.objects.new(name + '_obj', mesh_data)\n")
			
			# Write the code to link the mesh object to the current collection
			f.write("\tbpy.context.collection.objects.link(mesh_obj)\n")
			f.write("\tif parent is not None:\n\t\tmesh_obj.parent = parent\n")

			# Write the code to smooth faces
			f.write("\tfor poly in mesh_obj.data.polygons:\n")
			f.write("\t\tpoly.use_smooth = True\n")

			# Write edge split modifier if applicable
			for modifier in obj.modifiers:
				if isinstance(modifier, bpy.types.EdgeSplitModifier):
					f.write("\tmodifier = mesh_obj.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')\n")
					f.write(f"\tmodifier.split_angle = {modifier.split_angle}\n")
					f.write(f"\tmodifier.use_edge_angle = {modifier.use_edge_angle}\n")
					f.write(f"\tmodifier.use_edge_sharp = {modifier.use_edge_sharp}\n")
			
			f.write(f"\ncreate_mesh('new_mesh', None)\n")
		
		self.report({'INFO'}, "Python script for '" + obj.name + "' generated Successfully!")
		return {'FINISHED'}

class Archean_PreviewVoxelGrid(bpy.types.Operator):
	bl_idname = "object.preview_voxel_grid"
	bl_label = "Preview Voxel Grid"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		# Constants
		CUBE_SIZE = 0.125  # 12.5cm half-width (25cm full diameter)
		GRID_SPACING = 0.25  # 25cm spacing

		def delete_existing_object(name):
			"""If an object with the given name exists, delete it and all of its children."""
			if name in bpy.data.objects:
				# Deselect all
				bpy.ops.object.select_all(action='DESELECT')
				# Select target object and its children
				bpy.data.objects[name].select_set(True)
				for obj in bpy.data.objects[name].children:
					obj.select_set(True)
				# Delete selected
				bpy.ops.object.delete()

		def create_empty_cube(name, loc):
			"""Create an Empty Cube at the given location."""
			bpy.ops.object.empty_add(type='CUBE', location=loc)
			context.object.empty_display_size = CUBE_SIZE
			context.object.name = name

		def create_parent_object(name):
			"""Create a new object to act as a parent for all the cubes."""
			bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))
			context.object.name = name

		# Get current entity object
		obj = get_current_object(context)
		assert obj is not None, "No object selected"
		entityObj = get_entity_obj(obj)
		assert entityObj is not None, "Not a G4D Entity"

		# Compute bounds within entity's local space
		bbox = get_recursive_collider_bounds(entityObj)
		inv_matrix_world = entityObj.matrix_world.inverted()
		bbox['min'] = inv_matrix_world @ bbox['min']
		bbox['max'] = inv_matrix_world @ bbox['max']
		bbox_center = (bbox['min'] + bbox['max']) / 2
		bbox_size = bbox['max'] - bbox['min']

		# Compute number of cubes in each direction
		count_x = max(1, round(abs(bbox_size.x) / GRID_SPACING))
		count_y = max(1, math.floor(abs(bbox_size.y) / GRID_SPACING))
		count_z = max(1, round(abs(bbox_size.z) / GRID_SPACING))

		occupancy_size_x = count_x * GRID_SPACING
		occupancy_size_y = count_y * GRID_SPACING
		occupancy_size_z = count_z * GRID_SPACING

		# Delete existing g4d-voxel-grid and its children
		delete_existing_object('g4d-voxel-grid')

		# Create parent object
		create_parent_object('g4d-voxel-grid')
		parent = bpy.data.objects['g4d-voxel-grid']

		# Generate grid of empty cubes
		for i in range(count_x):
			for j in range(count_y):
				for k in range(count_z):
					pos_x = bbox_center.x - occupancy_size_x/2 + i * GRID_SPACING + CUBE_SIZE
					pos_y = bbox['min'].y + j * GRID_SPACING + CUBE_SIZE
					pos_z = bbox_center.z - occupancy_size_z/2 + k * GRID_SPACING + CUBE_SIZE
					create_empty_cube(f'Cube_{i}_{j}_{k}', (pos_x, pos_y, pos_z))
					cube = bpy.data.objects[f'Cube_{i}_{j}_{k}']
					cube.parent = parent
		parent.matrix_world = entityObj.matrix_world.copy()
		
		# re-select last active obj
		bpy.ops.object.select_all(action='DESELECT')
		context.view_layer.objects.active = obj
		obj.select_set(True)
		
		self.report({'INFO'}, "Voxel Grid for '" + entityObj.name + "' generated Successfully!")
		return {'FINISHED'}

class Archean_FixObjects(bpy.types.Operator):
	bl_idname = "object.archean_fix"
	bl_label = "Fix Now!"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		obj = get_current_object(context)
		assert obj is not None, "No object selected"
		entityObj = get_entity_obj(obj)
		apply_scale_to_obj_and_children(entityObj)
		bpy.ops.object.select_all(action='DESELECT')
		obj.select_set(True)
		bpy.context.view_layer.objects.active = obj
		
		self.report({'INFO'}, "Entity '" + entityObj.name + "' Fixed Successfully!")
		return {'FINISHED'}

class Archean_GenerateThumbnail(bpy.types.Operator):
	bl_idname = "object.archean_generate_thumbnail"
	bl_label = "Generate Thumbnail"
	
	def execute(self, context):
		if context.object is not None:
			image_path = render_and_save_thumbnail(context.object)

			# Load the image to memory
			bpy.ops.image.open(filepath=image_path)
			
			# Try to find an existing Image Editor space to display the image
			for area in context.screen.areas:
				if area.type == 'IMAGE_EDITOR':
					for space in area.spaces:
						if space.type == 'IMAGE_EDITOR':
							space.image = bpy.data.images.load(image_path)
							break

		return {'FINISHED'}

class Archean_ExportObject(bpy.types.Operator):
	bl_idname = "object.archean_export"
	bl_label = "Export this Entity and Save"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		if bpy.data.filepath == "":
			self.report({'ERROR'}, "Blend file not saved yet, please save the project first.")
			return {'CANCELLED'}
		
		# Get active object
		obj = get_current_object(context)
		if obj is None:
			self.report({'ERROR'}, "No object selected!")
			return {'CANCELLED'}
		
		# Get parent entity (or self)
		entityObj = get_entity_obj(obj)
		if entityObj is None:
			self.report({'ERROR'}, "Selected object is not part of an entity")
			return {'CANCELLED'}
		
		cleanup_unused_renderable_export_files(entityObj)
		
		# Prepare for export
		bpy.ops.object.select_all(action='DESELECT')
		prepare_objects_for_export(entityObj)
		bpy.context.view_layer.objects.active = entityObj
		
		# Export now
		export_gltf(entityObj, bpy.path.abspath("//") + "/" + entityObj.name)
		export_ini(entityObj, bpy.path.abspath("//") + "/" + entityObj.name)
		
		# Export Thumbnail
		# render_and_save_thumbnail(entityObj)
		
		# Select back the current object
		bpy.ops.object.select_all(action='DESELECT')
		obj.select_set(True)
		bpy.context.view_layer.objects.active = obj
		
		self.report({'INFO'}, "Entity '" + entityObj.name + "' Exported Successfully!")
		bpy.ops.wm.save_mainfile()
		return {'FINISHED'}

def Archean_UpdateObject_JointPreviewAngularLowX(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_X = -1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularHighX(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_X = 1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularNeutralX(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_X = 0
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularLowY(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_Y = -1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularHighY(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_Y = 1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularNeutralY(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_Y = 0
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularLowZ(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_Z = -1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularHighZ(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_Z = 1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewAngularNeutralZ(self, context):
	context.object.G4D_JOINT_PREVIEW_ANGULAR_Z = 0
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearLowX(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_X = -1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearHighX(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_X = 1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearNeutralX(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_X = 0
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearLowY(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_Y = -1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearHighY(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_Y = 1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearNeutralY(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_Y = 0
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearLowZ(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_Z = -1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearHighZ(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_Z = 1
	Archean_UpdateObject(self, context)
def Archean_UpdateObject_JointPreviewLinearNeutralZ(self, context):
	context.object.G4D_JOINT_PREVIEW_LINEAR_Z = 0
	Archean_UpdateObject(self, context)

def Archean_UpdateObject(self, context):
	update_obj_joint_preview(context.object)
def update_obj_joint_preview(obj):
	
	jointPreviewAngularX = getattr(obj, "G4D_JOINT_PREVIEW_ANGULAR_X", 0)
	jointPreviewAngularY = getattr(obj, "G4D_JOINT_PREVIEW_ANGULAR_Y", 0)
	jointPreviewAngularZ = getattr(obj, "G4D_JOINT_PREVIEW_ANGULAR_Z", 0)
	jointPreviewLinearX = getattr(obj, "G4D_JOINT_PREVIEW_LINEAR_X", 0)
	jointPreviewLinearY = getattr(obj, "G4D_JOINT_PREVIEW_LINEAR_Y", 0)
	jointPreviewLinearZ = getattr(obj, "G4D_JOINT_PREVIEW_LINEAR_Z", 0)
	
	rotation = obj.matrix_local.to_euler("XYZ")
	position = obj.matrix_local.translation.copy()
	
	if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_X", False):
		if jointPreviewAngularX == -1:
			rotation.x = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_X_LOW", 0.0))
		elif jointPreviewAngularX == 1:
			rotation.x = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_X_HIGH", 0.0))
		else:
			rotation.x = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_X_NEUTRAL", 0.0))
	if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_Y", False):
		if jointPreviewAngularY == -1:
			rotation.y = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_Y_LOW", 0.0))
		elif jointPreviewAngularY == 1:
			rotation.y = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_Y_HIGH", 0.0))
		else:
			rotation.y = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_Y_NEUTRAL", 0.0))
	if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_Z", False):
		if jointPreviewAngularZ == -1:
			rotation.z = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_Z_LOW", 0.0))
		elif jointPreviewAngularZ == 1:
			rotation.z = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_Z_HIGH", 0.0))
		else:
			rotation.z = math.radians(getattr(obj, "G4D_JOINT_ANGULAR_Z_NEUTRAL", 0.0))
	if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_X", False):
		if jointPreviewLinearX == -1:
			position.x = getattr(obj, "G4D_JOINT_LINEAR_X_LOW", 0.0)
		elif jointPreviewLinearX == 1:
			position.x = getattr(obj, "G4D_JOINT_LINEAR_X_HIGH", 0.0)
		else:
			position.x = getattr(obj, "G4D_JOINT_LINEAR_X_NEUTRAL", 0.0)
	if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_Y", False):
		if jointPreviewLinearY == -1:
			position.y = getattr(obj, "G4D_JOINT_LINEAR_Y_LOW", 0.0)
		elif jointPreviewLinearY == 1:
			position.y = getattr(obj, "G4D_JOINT_LINEAR_Y_HIGH", 0.0)
		else:
			position.y = getattr(obj, "G4D_JOINT_LINEAR_Y_NEUTRAL", 0.0)
	if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_Z", False):
		if jointPreviewLinearZ == -1:
			position.z = getattr(obj, "G4D_JOINT_LINEAR_Z_LOW", 0.0)
		elif jointPreviewLinearZ == 1:
			position.z = getattr(obj, "G4D_JOINT_LINEAR_Z_HIGH", 0.0)
		else:
			position.z = getattr(obj, "G4D_JOINT_LINEAR_Z_NEUTRAL", 0.0)
	
	obj.matrix_local = Matrix.Translation(position) @ rotation.to_matrix().to_4x4()

def Archean_ResetObjectJointPreview(self, context):
	reset_obj_joint_preview(context.object)
def reset_obj_joint_preview(obj):
	obj.G4D_JOINT_PREVIEW_ANGULAR_X = 0
	obj.G4D_JOINT_PREVIEW_ANGULAR_Y = 0
	obj.G4D_JOINT_PREVIEW_ANGULAR_Z = 0
	obj.G4D_JOINT_PREVIEW_LINEAR_X = 0
	obj.G4D_JOINT_PREVIEW_LINEAR_Y = 0
	obj.G4D_JOINT_PREVIEW_LINEAR_Z = 0
	update_obj_joint_preview(obj)

##########################################################################################################################

def create_material(name, metallic, roughness, hue, saturation, value, alpha):
	mat = bpy.data.materials.get(name)
	if mat is None:
		mat = bpy.data.materials.new(name=name)
		mat.use_nodes = True
		setup_material_nodes(mat, metallic, roughness, hue, saturation, value, alpha)
	else:
		mat.use_nodes = True
		setup_material_nodes(mat, metallic, roughness, hue, saturation, value, alpha)
	
	return mat

def setup_material_nodes(mat, metallic, roughness, hue, saturation, value, alpha):
	nodes = mat.node_tree.nodes
	nodes.clear()
	
	# Create principled BSDF node
	principled = nodes.new('ShaderNodeBsdfPrincipled')
	principled.inputs['Metallic'].default_value = metallic 
	principled.inputs['Roughness'].default_value = roughness
	
	# Convert HSV to RGB
	import colorsys
	rgb = colorsys.hsv_to_rgb(hue, saturation, value)
	principled.inputs['Base Color'].default_value = (rgb[0], rgb[1], rgb[2], alpha)
	principled.location = (0, 0)
	
	# Create output node
	output = nodes.new('ShaderNodeOutputMaterial')
	output.location = (300, 0)
	
	# Link nodes
	links = mat.node_tree.links
	links.new(principled.outputs[0], output.inputs[0])

class Archean_CreateMaterials(bpy.types.Operator):
	bl_idname = "object.archean_create_materials"
	bl_label = "Create Default Materials"
	bl_description = "Create default Archean materials"
	
	def execute(self, context):
		# List of materials to create: (name, metallic, roughness, hue, saturation, value, alpha)
		materials = [
			("body", 1, 1, 0, 0, 0.03, 1),
			("color1", 0, 0, 0, 0, 1, 1), 
			("color2", 1, 1, 0, 0, 0.8, 1),
			("data-connector", 1, 1, 0.65, 1, 0.5, 1),
			("lv-connector", 1, 1, 0, 1, 0.5, 1),
			("hv-connector", 1, 1, 0.05, 1, 0.5, 1),
			("fluid-connector", 1, 1, 0, 0, 0.5, 1),
			("h2",1,1,0.15,0.8,0.8,1),
			("o2",1,1,0.5,0.5,1,1),
			("h2o",1,1,0.65,0.8,0.8,1),
			("ch4",1,1,0,0.7,0.5,1)
		]
		
		# Create each material
		for params in materials:
			create_material(*params)
			
		self.report({'INFO'}, "Default materials created")
		return {'FINISHED'}

class Archean_CreateMesh(bpy.types.Operator):
	bl_idname = "object.archean_create_mesh"
	bl_label = "Create Mesh"
	bl_options = {'REGISTER', 'UNDO'}
	mesh_type: bpy.props.StringProperty()

	def execute(self, context):
		obj = context.object
		
		# Delete if already exists
		bpy.ops.object.select_all(action='DESELECT')
		for child in obj.children:
			if child.name.startswith(self.mesh_type + '_obj'):
				child.select_set(True)
		bpy.ops.object.delete()
		
		# Create new mesh
		switcher = {
			'data_adapter': create_data_adapter_mesh,
			'lowvoltage_adapter': create_lowvoltage_adapter_mesh,
			'highvoltage_adapter': create_highvoltage_adapter_mesh,
			'fluid_adapter': create_fluid_adapter_mesh,
			'item_adapter': create_item_adapter_mesh,
		}
		func = switcher.get(self.mesh_type, lambda: print("Invalid mesh type"))
		func(self.mesh_type, obj)
		
		# Select back the parent
		bpy.ops.object.select_all(action='DESELECT')
		context.view_layer.objects.active = obj
		obj.select_set(True)
		
		return {'FINISHED'}

##########################################################################################################################

# Export INI
def export_object_and_children_ini(config, obj, file_path):
	
	# ENTITY
	if getattr(obj, "G4D_IS_ENTITY", False):
		add_section_to_config(config, "ENTITY " + obj.name)
		add_property_to_config(config, "ENTITY " + obj.name, "mass", [getattr(obj, "G4D_ENTITY_MASS", 0.0)])
		add_property_to_config(config, "ENTITY " + obj.name, "airtight", getattr(obj, "G4D_ENTITY_AIRTIGHT", False))
		add_property_to_config(config, "ENTITY " + obj.name, "base_planes", "-Y -Z" if getattr(obj, "G4D_ENTITY_BASE_PLANE_IS_MINUS_Y", False) else "-Z -Y")
		
	else:
		parent = get_parent_obj(obj)
	
		# RENDERABLE
		if getattr(obj, "G4D_IS_RENDERABLE", False):
			add_parent_to_config(config, "RENDERABLE " + obj.name, parent)
			add_pos_and_rot_to_config(config, "RENDERABLE " + obj.name, obj)
			if getattr(obj, "G4D_RENDERABLE_EXPORT_SHARP_EDGES", False):
				assert obj.data is not None, "No mesh data for object '" + obj.name + "'"
				sharp_edges = []
				for edge in obj.data.edges:
					if edge.use_edge_sharp:
						sharp_edges.append((edge.vertices[0], edge.vertices[1]))
				if sharp_edges:
					assert len(obj.data.vertices) <= 65535, "Too many vertices to export sharp edges for object '" + obj.name + "'"
					with open(file_path + '.' + obj.name + '.vertices.bin', 'wb') as f:
						for v in obj.data.vertices:
							f.write(struct.pack('f', v.co.x))
							f.write(struct.pack('f', v.co.y))
							f.write(struct.pack('f', v.co.z))
					with open(file_path + '.' + obj.name + '.sharp_edges.bin', 'wb') as f:
						for v0, v1 in sharp_edges:
							f.write(struct.pack('H', v0))
							f.write(struct.pack('H', v1))
		
		# JOINT
		if getattr(obj, "G4D_IS_JOINT", False):
			add_parent_to_config(config, "JOINT " + obj.name, parent)
			add_pos_and_rot_to_config(config, "JOINT " + obj.name, obj)
		
			if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_X", False):
				add_property_to_config(config, "JOINT " + obj.name, "angular_x", [getattr(obj, "G4D_JOINT_ANGULAR_X_LOW", 0.0), getattr(obj, "G4D_JOINT_ANGULAR_X_NEUTRAL", 0.0), getattr(obj, "G4D_JOINT_ANGULAR_X_HIGH", 0.0)])
			if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_Y", False):
				add_property_to_config(config, "JOINT " + obj.name, "angular_y", [getattr(obj, "G4D_JOINT_ANGULAR_Y_LOW", 0.0), getattr(obj, "G4D_JOINT_ANGULAR_Y_NEUTRAL", 0.0), getattr(obj, "G4D_JOINT_ANGULAR_Y_HIGH", 0.0)])
			if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_Z", False):
				add_property_to_config(config, "JOINT " + obj.name, "angular_z", [getattr(obj, "G4D_JOINT_ANGULAR_Z_LOW", 0.0), getattr(obj, "G4D_JOINT_ANGULAR_Z_NEUTRAL", 0.0), getattr(obj, "G4D_JOINT_ANGULAR_Z_HIGH", 0.0)])
			if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_X", False):
				add_property_to_config(config, "JOINT " + obj.name, "linear_x", [getattr(obj, "G4D_JOINT_LINEAR_X_LOW", 0.0), getattr(obj, "G4D_JOINT_LINEAR_X_NEUTRAL", 0.0), getattr(obj, "G4D_JOINT_LINEAR_X_HIGH", 0.0)])
			if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_Y", False):
				add_property_to_config(config, "JOINT " + obj.name, "linear_y", [getattr(obj, "G4D_JOINT_LINEAR_Y_LOW", 0.0), getattr(obj, "G4D_JOINT_LINEAR_Y_NEUTRAL", 0.0), getattr(obj, "G4D_JOINT_LINEAR_Y_HIGH", 0.0)])
			if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_Z", False):
				add_property_to_config(config, "JOINT " + obj.name, "linear_z", [getattr(obj, "G4D_JOINT_LINEAR_Z_LOW", 0.0), getattr(obj, "G4D_JOINT_LINEAR_Z_NEUTRAL", 0.0), getattr(obj, "G4D_JOINT_LINEAR_Z_HIGH", 0.0)])
		
		# TARGET
		if getattr(obj, "G4D_IS_TARGET", False):
			add_parent_to_config(config, "TARGET " + obj.name, parent)
			add_pos_and_rot_to_config(config, "TARGET " + obj.name, obj)
		
		# CAMERA
		if getattr(obj, "G4D_IS_CAMERA", False):
			add_parent_to_config(config, "CAMERA " + obj.name, parent)
			add_pos_and_rot_to_config(config, "CAMERA " + obj.name, obj)
			if type(obj.data) == bpy.types.Camera:
				obj.G4D_CAMERA_ZNEAR = obj.data.clip_start
			add_property_to_config(config, "CAMERA " + obj.name, "znear", [getattr(obj, "G4D_CAMERA_ZNEAR", 0.0)])
		
		# COLLIDER
		if getattr(obj, "G4D_IS_COLLIDER", False):
			add_parent_to_config(config, "COLLIDER " + obj.name, parent)
			add_pos_and_rot_to_config(config, "COLLIDER " + obj.name, obj)
			bb = get_bounding_box(obj)
			assert bb is not None, "Invalid COLLIDER configuration for object '" + obj.name + "'"
			add_property_to_config(config, "COLLIDER " + obj.name, "is_build_block", getattr(obj, "G4D_IS_BUILD_BLOCK", False))
			add_property_to_config(config, "COLLIDER " + obj.name, "box_min", bb['min'])
			add_property_to_config(config, "COLLIDER " + obj.name, "box_max", bb['max'])
			if len(obj.data.vertices) <= 8:
				vertices = []
				for v in obj.data.vertices:
					vertices += [v.co.x, v.co.y, v.co.z]
				add_property_to_config(config, "COLLIDER " + obj.name, "vertices", vertices)
				assert len(obj.data.polygons) <= 12, "Invalid COLLIDER configuration for object '" + obj.name + "'"
				faces_normal = []
				triangles = []
				for p in obj.data.polygons:
					faces_normal += [p.normal.x, p.normal.y, p.normal.z]
					if len(p.vertices) == 3:
						triangles += [p.vertices[0], p.vertices[1], p.vertices[2]]
					elif len(p.vertices) == 4:
						triangles += [p.vertices[0], p.vertices[1], p.vertices[2]]
						triangles += [p.vertices[0], p.vertices[2], p.vertices[3]]
				add_property_to_config(config, "COLLIDER " + obj.name, "faces_normal", faces_normal)
				add_property_to_config(config, "COLLIDER " + obj.name, "triangles", triangles)
		
		# ADAPTER
		if getattr(obj, "G4D_IS_ADAPTER", False):
			add_parent_to_config(config, "ADAPTER " + obj.name, parent)
			adapterType = getattr(obj, "G4D_ADAPTER_TYPE", "").strip()
			assert adapterType != "", "Adapter type must not be an empty string in object '" + obj.name + "'"
			add_property_to_config(config, "ADAPTER " + obj.name, "type", adapterType)
			add_pos_and_rot_to_config(config, "ADAPTER " + obj.name, obj)
	
	# Do this recursively for children
	for child in obj.children:
		export_object_and_children_ini(config, child, file_path)

# Draw Panel
class Archean_Panel(bpy.types.Panel):
	bl_idname = "OBJECT_PT_archean"
	bl_label = "Archean"
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_category = "Archean"

	def draw(self, context):
		obj = context.object
		if obj is not None:
			layout = self.layout
			layout.operator("object.archean_export")
			
			if obj.data is not None and obj.type == 'MESH':
				layout.operator("object.write_mesh_as_python")
			
			row = layout.row()
			row.label(text=obj.name, icon="OBJECT_DATA")
			
			if should_fix_object_or_children(get_entity_obj(obj)):
				box = layout.box()
				box.alert = True
				box.row().label(text="Some objects are scaled, they should be applied", icon='ERROR')
				box.row().operator("object.archean_fix")

			# ENTITY
			box = layout.box()
			box.row().prop(obj, "G4D_IS_ENTITY")
			if getattr(obj, "G4D_IS_ENTITY", False):
				box.row().prop(obj, "G4D_ENTITY_MASS")
				box.row().prop(obj, "G4D_ENTITY_AIRTIGHT")
				box.row().prop(obj, "G4D_ENTITY_BASE_PLANE_IS_MINUS_Y")
				# box.row().prop(obj, "G4D_ENTITY_GLTF_EXPORT_COLORS")
				box.row().prop(obj, "G4D_ENTITY_GLTF_EXPORT_TEXCOORDS")
				box.row().operator("object.archean_generate_thumbnail")
				# box.row().operator("object.preview_voxel_grid") # THIS IS BROKEN AT THE MOMENT...
				return
			
			# RENDERABLE
			box = layout.box()
			box.row().prop(obj, "G4D_IS_RENDERABLE")
			if getattr(obj, "G4D_IS_RENDERABLE", False):
				if obj.data is None:
					box.alert = True
					box.row().label(text="ERROR: this object has no mesh", icon='ERROR')
				else:
					add_pos_and_rot_to_panel(box, obj)
					box.row().prop(obj, "G4D_RENDERABLE_EXPORT_SHARP_EDGES")
					if getattr(obj, "G4D_RENDERABLE_EXPORT_SHARP_EDGES", False):
						
						if len(obj.data.vertices) > 65535:
							box.alert = True
							box.row().label(text="Too many vertices to export sharp edges", icon='ERROR')
						else:
							box.row().label(text="Nb Vertices: " + str(len(obj.data.vertices)))
							nbSharpEdges = 0
							for edge in obj.data.edges:
								if edge.use_edge_sharp:
									nbSharpEdges += 1
							box.row().label(text="Nb sharp edges indices: " + str(nbSharpEdges))
							if nbSharpEdges == 0:
								box.row().label(text="No sharp edges detected; export will skip them.", icon='INFO')
			
			# JOINT
			box = layout.box()
			box.row().prop(obj, "G4D_IS_JOINT")
			if getattr(obj, "G4D_IS_JOINT", False):
				add_pos_and_rot_to_panel(box, obj)
				innerbox = box.box()
				innerbox.row().prop(obj, "G4D_JOINT_ENABLE_ANGULAR_X")
				if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_X", False):
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_X_LOW")
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_X_NEUTRAL")
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_X_HIGH")
				innerbox = box.box()
				innerbox.row().prop(obj, "G4D_JOINT_ENABLE_ANGULAR_Y")
				if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_Y", False):
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_Y_LOW")
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_Y_NEUTRAL")
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_Y_HIGH")
				innerbox = box.box()
				innerbox.row().prop(obj, "G4D_JOINT_ENABLE_ANGULAR_Z")
				if getattr(obj, "G4D_JOINT_ENABLE_ANGULAR_Z", False):
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_Z_LOW")
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_Z_NEUTRAL")
					innerbox.row().prop(obj, "G4D_JOINT_ANGULAR_Z_HIGH")
				innerbox = box.box()
				innerbox.row().prop(obj, "G4D_JOINT_ENABLE_LINEAR_X")
				if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_X", False):
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_X_LOW")
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_X_NEUTRAL")
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_X_HIGH")
				innerbox = box.box()
				innerbox.row().prop(obj, "G4D_JOINT_ENABLE_LINEAR_Y")
				if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_Y", False):
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_Y_LOW")
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_Y_NEUTRAL")
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_Y_HIGH")
				innerbox = box.box()
				innerbox.row().prop(obj, "G4D_JOINT_ENABLE_LINEAR_Z")
				if getattr(obj, "G4D_JOINT_ENABLE_LINEAR_Z", False):
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_Z_LOW")
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_Z_NEUTRAL")
					innerbox.row().prop(obj, "G4D_JOINT_LINEAR_Z_HIGH")
			
			# TARGET
			box = layout.box()
			box.row().prop(obj, "G4D_IS_TARGET")
			if getattr(obj, "G4D_IS_TARGET", False):
				add_pos_and_rot_to_panel(box, obj)
				
			# CAMERA
			box = layout.box()
			box.row().prop(obj, "G4D_IS_CAMERA")
			if getattr(obj, "G4D_IS_CAMERA", False):
				add_pos_and_rot_to_panel(box, obj)
				if type(obj.data) == bpy.types.Camera:
					box.row().label(text="Effective znear: " + f"{obj.data.clip_start:.2f}")
				else:
					box.row().prop(obj, "G4D_CAMERA_ZNEAR")
				
			# COLLIDER
			box = layout.box()
			box.row().prop(obj, "G4D_IS_COLLIDER")
			if getattr(obj, "G4D_IS_COLLIDER", False):
				box.row().prop(obj, "G4D_IS_BUILD_BLOCK")
				bb = get_bounding_box(obj)
				if bb is None:
					box.alert = True
					box.row().label(text="ERROR: this object has no mesh", icon='ERROR')
				else:
					add_pos_and_rot_to_panel(box, obj)
					# box.row().label(text="Effective dimensions: " + f"{obj.dimensions.x:.2f}, {obj.dimensions.y:.2f}, {obj.dimensions.z:.2f}")
					box.row().label(text="Effective bound min: " + f"{bb['min'].x:.2f}, {bb['min'].y:.2f}, {bb['min'].z:.2f}")
					box.row().label(text="Effective bound max: " + f"{bb['max'].x:.2f}, {bb['max'].y:.2f}, {bb['max'].z:.2f}")
					if len(obj.data.vertices) > 8:
						box.row().label(text="WARNING: this object has more than 8 vertices, its collider will be exported as a box", icon='ERROR')
					else:
						box.row().label(text="Number of vertices: " + f"{len(obj.data.vertices)}")
						box.row().label(text="Number of faces: " + f"{len(obj.data.polygons)}")
			
			# ADAPTER
			box = layout.box()
			box.row().prop(obj, "G4D_IS_ADAPTER")
			if getattr(obj, "G4D_IS_ADAPTER", False):
				box.row().prop(obj, "G4D_ADAPTER_TYPE")
				add_pos_and_rot_to_panel(box, obj)
				if getattr(obj, "G4D_ADAPTER_TYPE", "") != "":
					box.row().operator("object.archean_create_mesh").mesh_type = getattr(obj, "G4D_ADAPTER_TYPE", "") + "_adapter"
				
		layout.operator("object.archean_create_materials")


# Define Types
def register():
	bpy.utils.register_class(Archean_ExportObject)
	bpy.utils.register_class(Archean_FixObjects)
	bpy.utils.register_class(Archean_Panel)
	bpy.utils.register_class(Archean_GenerateThumbnail)
	bpy.utils.register_class(Archean_CreateMaterials)
	bpy.utils.register_class(Archean_PreviewVoxelGrid)
	bpy.utils.register_class(Archean_WriteMeshAsPython)
	bpy.utils.register_class(Archean_CreateMesh)
	bpy.utils.register_class(Archean_AddNewEntity)
	bpy.types.VIEW3D_MT_add.append(add_archean_entity_button)
	
	# ENTITY
	bpy.types.Object.G4D_IS_ENTITY = bpy.props.BoolProperty(
		name="Is Entity Root",
		description="Flags this object as the Entity's root in Archean",
		default=False
	)
	bpy.types.Object.G4D_ENTITY_MASS = bpy.props.FloatProperty(name="Mass (kg)",description="Base mass of this entity (when empty) in kilograms",default=10.0)
	bpy.types.Object.G4D_ENTITY_AIRTIGHT = bpy.props.BoolProperty(name="Airtight",description="Whether this entity's bounding box is considered air tight and has no leaks",default=False)
	bpy.types.Object.G4D_ENTITY_BASE_PLANE_IS_MINUS_Y = bpy.props.BoolProperty(name="Base Plane is Minus Y",description="Use Minus Y as the base plane instead of Minus Z. This relates to how the item is placed by default on a build. While in game, pressing Shift will reverse that behaviour and the mousewheel will rotate it on that axis.", default=False)
	# bpy.types.Object.G4D_ENTITY_GLTF_EXPORT_COLORS = bpy.props.BoolProperty(name="Export Vertex Colors",description="Export vertex colors for all renderables in this entity, otherwise just the material color will be assumed in Archean. At this moment this setting can only be applied globally per entity due to a limitation in Blender's GLTF exporter plugin.", default=False)
	bpy.types.Object.G4D_ENTITY_GLTF_EXPORT_TEXCOORDS = bpy.props.BoolProperty(name="Export Vertex UVs",description="Export vertex uvs for all renderables in this entity. At this moment this setting can only be applied globally per entity due to a limitation in Blender's GLTF exporter plugin.", default=False)

	# RENDERABLE
	bpy.types.Object.G4D_IS_RENDERABLE = bpy.props.BoolProperty(
		name="Is Renderable",
		description="Flags this object as Renderable in Archean",
		default=False
	)
	bpy.types.Object.G4D_RENDERABLE_EXPORT_SHARP_EDGES = bpy.props.BoolProperty(
		name="Export Sharp Edges",
		description="Exports a list of vertex indices marking sharp edges for this renderable in Archean",
		default=False
	)

	# JOINT
	bpy.types.Object.G4D_IS_JOINT = bpy.props.BoolProperty(
		name="Is Joint",
		description="Flags this object as a Joint in Archean",
		default=False
	)
	jointLowDescription = "Joint value when the normalized target is 0.0"
	jointNeutralDescription = "Initial/default joint value"
	jointHighDescription = "Joint value when the normalized target is 1.0"
	bpy.types.Object.G4D_JOINT_PREVIEW_ANGULAR_X = bpy.props.IntProperty(name="Joint preview angular X",default=0)
	bpy.types.Object.G4D_JOINT_PREVIEW_ANGULAR_Y = bpy.props.IntProperty(name="Joint preview angular Y",default=0)
	bpy.types.Object.G4D_JOINT_PREVIEW_ANGULAR_Z = bpy.props.IntProperty(name="Joint preview angular Z",default=0)
	bpy.types.Object.G4D_JOINT_ENABLE_ANGULAR_X = bpy.props.BoolProperty(name="Enable Rotation in X axis",description="Enables this joint to rotate in its X axis",default=False)
	bpy.types.Object.G4D_JOINT_ANGULAR_X_LOW = bpy.props.FloatProperty(name="Low", description=jointLowDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularLowX)
	bpy.types.Object.G4D_JOINT_ANGULAR_X_NEUTRAL = bpy.props.FloatProperty(name="Neutral", description=jointNeutralDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularNeutralX)
	bpy.types.Object.G4D_JOINT_ANGULAR_X_HIGH = bpy.props.FloatProperty(name="High", description=jointHighDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularHighX)
	bpy.types.Object.G4D_JOINT_ENABLE_ANGULAR_Y = bpy.props.BoolProperty(name="Enable Rotation in Y axis",description="Enables this joint to rotate in its Y axis",default=False)
	bpy.types.Object.G4D_JOINT_ANGULAR_Y_LOW = bpy.props.FloatProperty(name="Low", description=jointLowDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularLowY)
	bpy.types.Object.G4D_JOINT_ANGULAR_Y_NEUTRAL = bpy.props.FloatProperty(name="Neutral", description=jointNeutralDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularNeutralY)
	bpy.types.Object.G4D_JOINT_ANGULAR_Y_HIGH = bpy.props.FloatProperty(name="High", description=jointHighDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularHighY)
	bpy.types.Object.G4D_JOINT_ENABLE_ANGULAR_Z = bpy.props.BoolProperty(name="Enable Rotation in Z axis",description="Enables this joint to rotate in its Z axis",default=False)
	bpy.types.Object.G4D_JOINT_ANGULAR_Z_LOW = bpy.props.FloatProperty(name="Low", description=jointLowDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularLowZ)
	bpy.types.Object.G4D_JOINT_ANGULAR_Z_NEUTRAL = bpy.props.FloatProperty(name="Neutral", description=jointNeutralDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularNeutralZ)
	bpy.types.Object.G4D_JOINT_ANGULAR_Z_HIGH = bpy.props.FloatProperty(name="High", description=jointHighDescription, default=0.0, update=Archean_UpdateObject_JointPreviewAngularHighZ)
	bpy.types.Object.G4D_JOINT_PREVIEW_LINEAR_X = bpy.props.IntProperty(name="Joint preview linear X",default=0)
	bpy.types.Object.G4D_JOINT_PREVIEW_LINEAR_Y = bpy.props.IntProperty(name="Joint preview linear Y",default=0)
	bpy.types.Object.G4D_JOINT_PREVIEW_LINEAR_Z = bpy.props.IntProperty(name="Joint preview linear Z",default=0)
	bpy.types.Object.G4D_JOINT_ENABLE_LINEAR_X = bpy.props.BoolProperty(name="Enable Translation in X axis",description="Enables this joint to translate in its X axis",default=False)
	bpy.types.Object.G4D_JOINT_LINEAR_X_LOW = bpy.props.FloatProperty(name="Low", description=jointLowDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearLowX)
	bpy.types.Object.G4D_JOINT_LINEAR_X_NEUTRAL = bpy.props.FloatProperty(name="Neutral", description=jointNeutralDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearNeutralX)
	bpy.types.Object.G4D_JOINT_LINEAR_X_HIGH = bpy.props.FloatProperty(name="High", description=jointHighDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearHighX)
	bpy.types.Object.G4D_JOINT_ENABLE_LINEAR_Y = bpy.props.BoolProperty(name="Enable Translation in Y axis",description="Enables this joint to translate in its Y axis",default=False)
	bpy.types.Object.G4D_JOINT_LINEAR_Y_LOW = bpy.props.FloatProperty(name="Low", description=jointLowDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearLowY)
	bpy.types.Object.G4D_JOINT_LINEAR_Y_NEUTRAL = bpy.props.FloatProperty(name="Neutral", description=jointNeutralDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearNeutralY)
	bpy.types.Object.G4D_JOINT_LINEAR_Y_HIGH = bpy.props.FloatProperty(name="High", description=jointHighDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearHighY)
	bpy.types.Object.G4D_JOINT_ENABLE_LINEAR_Z = bpy.props.BoolProperty(name="Enable Translation in Z axis",description="Enables this joint to translate in its Z axis",default=False)
	bpy.types.Object.G4D_JOINT_LINEAR_Z_LOW = bpy.props.FloatProperty(name="Low", description=jointLowDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearLowZ)
	bpy.types.Object.G4D_JOINT_LINEAR_Z_NEUTRAL = bpy.props.FloatProperty(name="Neutral", description=jointNeutralDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearNeutralZ)
	bpy.types.Object.G4D_JOINT_LINEAR_Z_HIGH = bpy.props.FloatProperty(name="High", description=jointHighDescription, default=0.0, update=Archean_UpdateObject_JointPreviewLinearHighZ)

	# TARGET
	bpy.types.Object.G4D_IS_TARGET = bpy.props.BoolProperty(
		name="Is Target",
		description="Flags this object as a Target in Archean",
		default=False
	)

	# CAMERA
	bpy.types.Object.G4D_IS_CAMERA = bpy.props.BoolProperty(
		name="Is Camera",
		description="Flags this object as a Camera in Archean (positions the camera relative to the player's head and defines the near field)",
		default=False
	)
	bpy.types.Object.G4D_CAMERA_ZNEAR = bpy.props.FloatProperty(name="Z Near", description="Clipping plane for the Near field of this camera, useful for making a head invisible for primary rays")

	# COLLIDER
	bpy.types.Object.G4D_IS_COLLIDER = bpy.props.BoolProperty(
		name="Is Collider",
		description="Flags this object as a Collider in Archean (will make a bounding box with the objet's dimensions and use it for collision detection)",
		default=False
	)
	bpy.types.Object.G4D_IS_BUILD_BLOCK = bpy.props.BoolProperty(
		name="Is Build Block",
		description="Flags this collider as a Build Block in Archean (gives the ability to add other blocks or components on top of it)",
		default=False
	)

	# ADAPTER
	bpy.types.Object.G4D_IS_ADAPTER = bpy.props.BoolProperty(
		name="Is Adapter",
		description="Flags this object as an Adapter in Archean (will use its precise position as the connection point and its rotation where +Z points in the direction of the connection)",
		default=False
	)
	bpy.types.Object.G4D_ADAPTER_TYPE = bpy.props.EnumProperty(
		name="Type",
		description="Specify the type of adapter - Two adapters can only connect if their type matches exactly",
		items=[
			('data', "data", "Data adapter"),
			('lowvoltage', "lowvoltage", "LowVoltage adapter"),
			('highvoltage', "highvoltage", "HighVoltage adapter"),
			('fluid', "fluid", "Fluid adapter"),
			('item', "item", "Item adapter"),
		],
		default='data'
	)

def unregister():
	bpy.utils.unregister_class(Archean_ExportObject)
	bpy.utils.unregister_class(Archean_FixObjects)
	bpy.utils.unregister_class(Archean_Panel)
	bpy.utils.unregister_class(Archean_GenerateThumbnail)
	bpy.utils.unregister_class(Archean_CreateMaterials)
	bpy.utils.unregister_class(Archean_PreviewVoxelGrid)
	bpy.utils.unregister_class(Archean_WriteMeshAsPython)
	bpy.utils.unregister_class(Archean_CreateMesh)
	bpy.utils.unregister_class(Archean_AddNewEntity)
	bpy.types.VIEW3D_MT_add.remove(add_archean_entity_button)

if __name__ == "__main__":
	register()
