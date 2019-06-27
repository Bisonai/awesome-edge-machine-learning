from style import h2
from style import newline


def renderer(fp, data, config):
   fp.write(data["logo"])
   newline(fp)
   newline(fp)
   fp.write(data["description"])
