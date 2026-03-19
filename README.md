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
    
