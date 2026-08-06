import os

from backend.app import create_app


app = create_app()


if __name__ == "__main__":
    # Loopback by default, not 0.0.0.0. Registering a dataset unpickles the uploaded file,
    # which executes code from it, so this server must not be reachable from the network
    # unless its operator deliberately says otherwise via IPD_HOST.
    app.run(
        host=os.environ.get("IPD_HOST", "127.0.0.1"),
        port=int(os.environ.get("IPD_PORT", "5000")),
        debug=True,
    )
