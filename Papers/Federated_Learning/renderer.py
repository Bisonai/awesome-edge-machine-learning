from style import li
from style import h2
from style import a
from style import p
from style import newline


def renderer(fp, data, config):
    year, month, day = data["date"].split("/")
    fp.write(h2(a([data["name"].strip(), data["url"].strip()]) + f", {year}/{month}"))
    fp.write(data["authors"])
    newline(fp)
    newline(fp)
    fp.write(p(data["abstract"]))
    newline(fp)
