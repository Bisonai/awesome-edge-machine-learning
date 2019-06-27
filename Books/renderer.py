from style import h3
from style import p
from style import a
from style import newline
from style import li


# TODO separate authors and add links
def renderer(fp, data, config):
    title = data["title"] if data["subtitle"] is None else f"{data['title']}: {data['subtitle']}"
    fp.write(h3(a([
        title,
        data["url"],
    ])))

    # Authors
    if len(data["authors"]) > 1:
        author = "Authors: "
    else:
        author = "Author: "

    author += ", ".join(data["authors"])
    li(fp, author)

    li(fp, f"Published: {data['published']}")
    newline(fp)
