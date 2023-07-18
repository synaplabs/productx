# Synap-Labs

## Steps to run the project

1. Clone the project: 
```
git clone https://github.com/synaplabs/productx.git
```

2. Create a local Python Virtual Environment and activate it (optional but recommended) (if you don't have virtualenv installed, run `pip install virtualenv`): 
```
python3 -m venv synap_labs_github_env
```
Mac/Linux:
```
source synap_labs_github_env/bin/activate
```
Windows:
```
synap_labs_github_env\Scripts\activate.bat
```

3. Install the project dependencies: 
```
pip install -r requirements.txt
```

4. Install wandb-addons for the tracking results:
```
git clone https://github.com/soumik12345/wandb-addons.git
pip install ./wandb-addons[prompts] openai wandb -qqq
```

4. Setup Streamlit secrets (environment variables):
    
    Create a folder named `.streamlit/` within the `streamlit_web` folder.

    Inside the `.streamlit/` folder, create a file named `secrets.toml` with the following content:
    ```
    OPENAI_API_KEY = "YOUR_API_KEY"
    WANDB_API_KEY = "YOUR_API_KEY" 
    ```

5. Setup Google Cloud Auth Credentials:
    
    Follow the steps listed here: https://googleapis.dev/python/google-api-core/latest/auth.html

6. Run the Streamlit app:
```
cd streamlit_web
streamlit run main.py
```