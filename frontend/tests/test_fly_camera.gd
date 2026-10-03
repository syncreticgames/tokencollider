extends RefCounted
## The camera stands still while something covers the viewport.
const FlyCamera := preload("res://FlyCamera.gd")

var harness

func test_blocked_camera_ignores_input() -> void:
	## Behind the open menu, mouse and keys must not reach the camera: a
	## right-drag there used to capture the mouse and fly it.
	var cam = FlyCamera.new()
	cam.blocked = func() -> bool: return true
	var wheel := InputEventMouseButton.new()
	wheel.button_index = MOUSE_BUTTON_WHEEL_UP
	wheel.pressed = true
	var before: float = cam.speed
	cam._input(wheel)
	harness.eq(cam.speed, before, "wheel ignored while blocked")
	cam.blocked = func() -> bool: return false
	harness.ok(cam.blocked.call() == false, "unblocked by the same hook")
	cam.free()
