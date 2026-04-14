# ESKF
Choose between estimation modes computed on the ASV:
- tracking of ROV only
- tracking of ROV and navigation of ROV

Runs 3 scenarios (for t+n GNSS is included in all): 
1. Bearing (azimuth and elevation) only
2. Bearing + range
3. Bearing + range + depth

Bearing and range is computed on the ASV by an USBL, depth needs to be sent acoustically from the ROV and is received by the USBL.

## Quick start

1. Install uv
    
    - macOS/Linux:
        
        ```
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
        
    - Windows (PowerShell):
        
        ```
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
        
2. Clone the repository
    
    ```
    git clone <your repo>
    cd <your repo>
    ```
    
3. Install dependencies
    
    ```
    uv sync
    ```
    
4. Run the project
    
    ```
    uv run python src/main.py
    ```
    
## Dataset generation for FGO comparison
This program is meant to be compared to a FGO solution to the same problem, thus they should both be run with the same dataset. As the ESKF already had a dataset generator, scripts were made to convert the dataset into ros2 bags fitting for the FGO.

To generate the correct ros2 bags ROS2 humble and the microampere interfaces are needed, and a docker container was made to simplify the process. The docker container autommatically sets up the ros2 environment, copies the ESKF files, clones blueboat interfaces and switches to the correct "microAmp/fgo_rov_tracking" branch. Ros2 is automatically sourced in the docker terminal as well. 

Opening docker (in terminal):
```
docker build -t eskf-humble:latest .
docker-compose up -d
docker exec -it eskf_humble bash 
```

Reopening in docker container migth be preferable to only opening the container in terminal.