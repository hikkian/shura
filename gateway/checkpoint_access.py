"""Keep native checkpoint mutations private to the gateway's lifecycle code."""
import posixpath
from urllib.parse import unquote, urlsplit


def private_checkpoint_handler(base):
    class PrivateCheckpoints(base):
        def handle_any(self):
            path = posixpath.normpath(unquote(urlsplit("/" + self.path.lstrip("/")).path)).lstrip("/")
            checkpoint_path = path == "slots" or path.startswith("slots/")
            if checkpoint_path and self.command not in ("GET", "HEAD", "OPTIONS"):
                # Do not leave an unread body on a reusable HTTP connection.
                self.close_connection = True
                return self.send_json(403, {
                    "error": "Checkpoint mutations are managed internally by the gateway",
                    "type": "private_checkpoint_endpoint",
                })
            return super().handle_any()

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = handle_any

    return PrivateCheckpoints
