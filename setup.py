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
    os.makedirs("models", exist_ok=True)
    os.makedirs("references", exist_ok=True)

    # create agent directories
    if version:
        model_path = os.path.join("models", f"{agent}_v{version}")
        reference_path = os.path.join("references", f"{agent}_v{version}")
    else:
        model_path = os.path.join("models", agent)
        reference_path = os.path.join("references", agent)

    os.makedirs(os.path.join(model_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(reference_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(reference_path, "videos"), exist_ok=True)
    os.makedirs(os.path.join(reference_path, "evaluation"), exist_ok=True)

    return True
