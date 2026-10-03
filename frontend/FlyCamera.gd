extends Camera3D
## Minimal free-fly camera: hold right mouse to look, WASD to move, Q/E down/up,
## Shift for a burst, mousewheel to raise/lower the base speed.
## Swap for your own controller when porting it in.

@export var speed := 12.0
@export var sensitivity := 0.003

@onready var status: Label = get_node_or_null("../UI/Status")
## True while something covers the viewport (Main wires it to the menu). The
## camera reads the mouse in _input, ahead of the menu, so without this a
## right-drag behind the open menu captured the mouse and WASD flew the camera.
var blocked: Callable = func() -> bool: return false

func _input(event: InputEvent) -> void:
	if blocked.call():
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		return
	if event is InputEventMouseButton and event.pressed and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
		if get_viewport().gui_get_focus_owner() != null:
			return  # let a focused control keep its scroll
		var factor := 1.2 if event.button_index == MOUSE_BUTTON_WHEEL_UP else 1.0 / 1.2
		speed = clampf(speed * factor, 0.5, 300.0)
		if status != null:
			status.text = "fly speed %.1f" % speed
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
		if event.pressed:
			# Take the keyboard back from the word box, else _process stays parked.
			get_viewport().gui_release_focus()
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if event.pressed else Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		# Drive angles directly: clamped pitch can't flip past vertical (which
		# mirror-reverses yaw/strafe), and roll can't accumulate.
		rotation.y -= event.relative.x * sensitivity
		rotation.x = clampf(rotation.x - event.relative.y * sensitivity,
			-PI / 2 + 0.01, PI / 2 - 0.01)
		rotation.z = 0.0

func _process(delta: float) -> void:
	if blocked.call() or get_viewport().gui_get_focus_owner() != null:
		return  # behind the menu, or typing in the word box: don't fly
	var dir := Vector3.ZERO
	if Input.is_key_pressed(KEY_W): dir -= global_transform.basis.z
	if Input.is_key_pressed(KEY_S): dir += global_transform.basis.z
	if Input.is_key_pressed(KEY_A): dir -= global_transform.basis.x
	if Input.is_key_pressed(KEY_D): dir += global_transform.basis.x
	if Input.is_key_pressed(KEY_Q): dir -= Vector3.UP
	if Input.is_key_pressed(KEY_E): dir += Vector3.UP
	if dir != Vector3.ZERO:
		var mult := 3.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0
		global_position += dir.normalized() * speed * mult * delta
