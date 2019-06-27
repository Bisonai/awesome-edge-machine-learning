from style import h3
from style import p
from style import a
from style import li
from style import newline
from utils import name2dir


# TODO extract Papers directory automatically
def renderer(fp, data, config):
    fp.write(h3(a([
        data["name"],
        config["url"] + "/tree/master/Papers/" + name2dir(data["name"]),
    ])))
    fp.write(p(data["description"]))
    fp.write("\n")


def renderer_subdir(fp, data, config):
    li(fp, a([data["name"].strip(), data["url"]]) + ". " + data["authors"].strip())
    newline(fp)
