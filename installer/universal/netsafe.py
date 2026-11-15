"""One shared URL opener for everything the installer downloads (manifest, engine archive, model): a redirect may only
lead to https, so an https address can never be bounced down to plain http."""
import urllib.error
import urllib.request


class InsecureRedirect(urllib.error.URLError):
    pass


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise InsecureRedirect(f"redirect to non-HTTPS URL refused: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(HttpsOnlyRedirect)


def urlopen(request, timeout=60):
    return _OPENER.open(request, timeout=timeout)
