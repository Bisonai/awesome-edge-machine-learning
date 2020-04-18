from style import li
from style import h3
from style import a


def renderer(fp, data, config):
    fp.write(h3(a([data["name"], data["link"]])))

    if data["description"] is not None:
        fp.write(data["description"])
        fp.write("\n")

    fp.write("\n")
