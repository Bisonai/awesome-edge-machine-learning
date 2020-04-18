from typing import List


def concatenate(text: List):
    return "".join(filter(lambda x: x is not None, text))


def li(fp, text):
    if isinstance(text, list):
        text = concatenate(text)

    fp.write("- " + text + "\n")


def lili(fp, text):
    """Second level of list items"""
    fp.write("\t")
    li(fp, text)


def h1(text):
    return "# " + text + "\n"


def h2(text):
    return "## " + text + "\n"


def h3(text):
    return "### " + text + "\n"


def h4(text):
    return "#### " + text + "\n"


def h5(text):
    return "##### " + text + "\n"


def h6(text):
    return "###### " + text + "\n"


def p(text):
    if text is None:
        return "\n"
    else:
        return str(text) + "\n"


def a(args: List):
    if not isinstance(args, list):
        args = [args]

    if len(args) == 1:
        src = args[0]
        if src is None:
            return ""
        else:
            return f"[{src}]({src})"
    if len(args) == 2:
        name = args[0]
        src = args[1]
        if name is None or src is None:
            return ""
        else:
            return f"[{name}]({src})"
    else:
        raise NotImplementedError


def newline(fp, iter=1):
    for _ in range(iter):
        fp.write("\n")
