from style import p
from style import a
from style import h3
from style import newline
from utils import name2link


def renderer(fp, data, config):
    fp.write(h3(a([
        data["name"],
        data["url"],
    ])))
    newline(fp)
    fp.write(p(data["description"]))
    newline(fp)
