from style import li
from style import h3
from style import a


def renderer(fp, data, config):
    fp.write(h3(data["name"]))

    if data["company"] is not None:
        li(fp, ["Company: ", data["company"]])

    if data["link"] is not None:
        li(fp, [
            a(data["link"]),
        ])
        fp.write("\n")

    if data["description"] is not None:
        fp.write(data["description"])
        fp.write("\n")

    fp.write("\n")
