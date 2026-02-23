# run.py
import sys
import os

# Fix Windows asyncio / ProactorEventLoop incompatibility
# Python 3.8+ sur Windows utilise ProactorEventLoop par défaut,
# incompatible avec certaines librairies (Twisted, gevent, Flask-dev).
# SelectorEventLoop = comportement Linux → résout les erreurs réseau silencieuses.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dash_bot.app import main

if __name__ == "__main__":
    main()
