"""Serve the browser UI and forward image tensors to local Triton."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

HOST = "127.0.0.1"
PORT = 8080
TRITON_URL = "localhost:8000"
MODEL_NAME = "plant_disease_model"
PAGE = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        content = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != "/infer":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            image = np.asarray(request["data"], dtype=np.float32)
            if image.size != 3 * 128 * 128:
                raise ValueError("Expected a 128x128 RGB image tensor")
            tensor = image.reshape(1, 3, 128, 128)
            client = InferenceServerClient(url=TRITON_URL)
            triton_input = InferInput("image", tensor.shape, "FP32")
            triton_input.set_data_from_numpy(tensor)
            response = client.infer(
                model_name=MODEL_NAME,
                inputs=[triton_input],
                outputs=[InferRequestedOutput("probabilities")],
            )
            scores = response.as_numpy("probabilities")[0].tolist()
            self._send_json(200, {"scores": scores})
        except Exception as error:
            self._send_json(500, {"error": str(error)})

    def _send_json(self, status, payload):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT} in your browser")
    print("Keep Triton running in its own terminal.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
