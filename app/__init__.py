"""CodeSentinel Flask application."""
import os

from flask import Flask, jsonify, render_template, request

from .config import Config
from .inference import Analyzer


def create_app(config_overrides=None, analyzer=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_overrides:
        app.config.update(config_overrides)
    # Reject oversized requests before Flask reads the body.
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_REQUEST_BYTES"]

    # Load the model ONCE at startup (never per request).
    app.extensions["analyzer"] = analyzer if analyzer is not None else Analyzer.from_config(app.config)

    from .routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_large(_e):
        msg = "The request is too large. Paste a single function (at most %d characters)." % app.config["MAX_INPUT_CHARS"]
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": msg}), 413
        return render_template("analyze.html", error=msg, code="", result=None,
                               examples=[], info=app.extensions["analyzer"].info()), 413

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Not found."}), 404
        return render_template("message.html", title="Page not found",
                               message="That page does not exist."), 404

    @app.errorhandler(500)
    def server_error(_e):
        # Never show a stack trace or file paths to the user.
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Internal error. The input was not analysed."}), 500
        return render_template("message.html", title="Something went wrong",
                               message="An internal error happened. The input was not analysed."), 500

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    return app
