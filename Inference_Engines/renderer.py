from style import li
from style import h3
from style import a


# TODO image: platform(iOS, Android), 32, 16, 8 bits, gpu/cpu acceleration
# TODO link to companies
def renderer(fp, data, config):
    fp.write(h3(data["name"]))

    if data["sourcecode"] is not None:
        li(fp, [
            "Source code: ",
            a(data["sourcecode"]),
        ])

    if data["documentation"] is not None:
        li(fp, [
            "Documentation: ",
            a(data["documentation"]),
        ])

    li(fp, data["company"])
    fp.write("\n")
