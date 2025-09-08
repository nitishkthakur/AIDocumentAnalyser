from flask import Flask, Response, render_template, request, stream_with_context
import time

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    q = request.args.get("q", "").strip()
    base = (
        "Thanks for your message. This is a demo streaming response from Jarvis Lite. "
        "It shows how text can appear progressively in the chat, just like an LLM would."
    )
    msg = f"You said: '{q}'. " + base if q else base

    def generate():
        yield "event: start\n" "data: started\n\n"
        for token in msg.split(" "):
            yield f"data: {token} \n\n"
            time.sleep(0.05)
        yield "event: done\n" "data: ok\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
