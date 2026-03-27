from __future__ import annotations

from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint

from .config import OPENAPI_PATH, SWAGGER_UI_PATH
from .openapi import build_openapi_spec
from .routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.config["OPENAPI_SPEC"] = build_openapi_spec()

    swagger_blueprint = get_swaggerui_blueprint(
        SWAGGER_UI_PATH,
        OPENAPI_PATH,
        config={"app_name": "Sign Language Speech API"},
    )
    app.register_blueprint(swagger_blueprint, url_prefix=SWAGGER_UI_PATH)

    register_routes(app)
    return app
