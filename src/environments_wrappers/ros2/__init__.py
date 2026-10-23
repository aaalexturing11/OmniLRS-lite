__author__ = "Antoine Richard"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"


def enable_ros2(simulation_app, bridge_name="foxy", **kwargs) -> None:
    """
    Enables ROS2 in the simulation.

    Args:
        simulation_app (SimulationApp): SimulationApp instance.
        **kwargs: Additional keyword arguments."""

    from isaacsim.core.utils.extensions import enable_extension

    # Isaac Sim 6.2 keeps isaacsim.ros2.bridge and adds isaacsim.ros2.core under it.
    # The distro (Humble or Jazzy) is chosen by the libraries bundled with that
    # Isaac build when ROS_DISTRO is unset. Both names enable the same extension.
    if bridge_name not in ("foxy", "humble", "jazzy"):
        raise ValueError("Bridge not supported, please choose between foxy, humble and jazzy.")
    enable_extension("isaacsim.ros2.core")
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("omni.kit.viewport.actions")

    simulation_app.update()
