# Plant Disease Classification with Triton

This repository contains the project notebook, exported ONNX model, Triton model repository, and a small local web page for testing inference.

## Run Triton

From PowerShell in this folder, start the Triton server:

```powershell
docker run --rm --name triton `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}\model_repository:/models" `
  nvcr.io/nvidia/tritonserver:25.08-py3 `
  tritonserver --model-repository=/models
```

Keep that terminal open.

## Run the upload page

In a second PowerShell window, install the Python dependencies if needed and start the local web app:

```powershell
python -m pip install "tritonclient[http]" opencv-python numpy
python .\web_app.py
```

Open <http://localhost:8080>, choose a JPG or PNG leaf image, and click **Run prediction**. The result is shown as a class index and scores; map the index using the class ordering used when the model was trained.

The command-line client is also available:

```powershell
python .\infer_triton.py "path\to\leaf.jpg"
```
