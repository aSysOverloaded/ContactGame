import os
import tempfile

# Must be set before the app modules are imported.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["COUNTDOWN_SECONDS"] = "0"
os.environ["CONTACT_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
