# Example code for CLIP Image Classification

## Setting up environment
After cloning, you can set up a local Python environment using venv:

```
python3 -m venv {insert name of your choice}
source {path to the venv}/bin/activate
```

### Install packages
`pip3 install -r requirements.txt`

## Serve CLIP locally to use as an API
`fastapi dev clip.py`

### Test the API (can insert this into any part of your project)
`python3 test-fastapi-clip.py`