import predictor as serve

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "serve" above to the
# new file.

app = serve.app
