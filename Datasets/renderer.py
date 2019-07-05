from style import h3
from style import p
from style import a
from style import newline


def renderer(fp, data, config):
    fp.write(h3(a([
        data["name"],
        data["url"],
    ])))
    fp.write(p(data["description"]))
    newline(fp)
