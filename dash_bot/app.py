# dash_bot/app.py
from dash import Dash
import dash_bootstrap_components as dbc

from .config import APP_TITLE
from .theme import apply_theme
from .ui.layout import build_layout
from .ui.callbacks import register_callbacks


def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    apply_theme(app, title=APP_TITLE)
    app.layout = build_layout()
    register_callbacks(app)
    return app


def main():
    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()