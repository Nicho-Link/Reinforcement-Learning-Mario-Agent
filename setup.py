import os


def create_directories(agent: str, version: str = None) -> bool:
    """setup directory structure

    Args:
        agent (str): name of the agent
        version (str, optional): if different versions of the agent will be created set version number. Defaults to None.

    Returns:
        bool: True if setup is done
    """
    # create base directories
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("references"):
        os.makedirs("references")

    # create agent directories
    if version:
        if not os.path.exists(f"models/{agent}_v{version}"):
            os.makedirs(f"models/{agent}_v{version}")
            os.makedirs(f"models/{agent}_v{version}/checkpoints")
        if not os.path.exists(f"references/{agent}_v{version}"):
            os.makedirs(f"references/{agent}_v{version}/images")
            os.makedirs(f"references/{agent}_v{version}/videos")
            os.makedirs(f"references/{agent}_v{version}/evaluation")
    else:
        if not os.path.exists(f"models/{agent}"):
            os.makedirs(f"models/{agent}")
            os.makedirs(f"models/{agent}/checkpoints")
        if not os.path.exists(f"references/{agent}"):
            os.makedirs(f"references/{agent}/images")
            os.makedirs(f"references/{agent}/videos")
            os.makedirs(f"references/{agent}/evaluation")

    return True
