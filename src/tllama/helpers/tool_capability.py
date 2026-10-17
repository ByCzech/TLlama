"""Offline tool-template indication for clients that gate requests on /show."""

from functools import lru_cache

from jinja2 import Environment, TemplateError, meta


def template_accepts_tools(template: object) -> bool:
    """Recognize an external tools input without executing untrusted Jinja.

    This is a template capability hint, not native parser/model certification.
    Unknown syntax is left unadvertised; it never forbids actual tool requests.
    Bound work/cache size for metadata supplied by arbitrary GGUF files.
    """
    if not isinstance(template, str) or not template or len(template) > 262144:
        return False
    return _accepts_tools(template)


@lru_cache(maxsize=32)
def _accepts_tools(template: str) -> bool:
    try:
        environment = Environment(extensions=["jinja2.ext.loopcontrols"])
        tree = environment.parse(template)
        return "tools" in meta.find_undeclared_variables(tree)
    except (TemplateError, RecursionError):
        # Native MiniJinja can understand syntax/filters Python Jinja cannot.
        return False
