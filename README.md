# Isaac Lab SES-V2-LSS Projects

## Overview
A project to train a arm-type robot to pick up and place the object well on a Conveyor belt using deep reinforcement learning.

## Installation

- Install Isaac Lab 5.0 version by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
```
git clone https://github.com/kimbring2/isaac_lab_ses_v2_lss.git
```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    python -m pip install -e source/ses_v2_lss

- Verify that the extension is correctly installed by:

    - Listing the available tasks:
        ```bash
        python scripts/list_envs.py
        ```

    - Running a training:

        ```bash
        python scripts/skrl/train.py --task SES-V2-LSS-v0
        ```

    - Running a testing
        ```
        python scripts/skrl/play.py --task SES-V2-LSS-v0 --num_envs 1
        ```
