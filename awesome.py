# AWESOME EDGE MACHINE LEARNING
# Bisonai, 2019
from pathlib import Path

from style import li
from style import lili
from style import h1
from style import h2
from style import a
from style import p
from style import newline
from utils import parse_yaml
from utils import name2dir
from utils import dir2name
from utils import name2link
from utils import find_subdirectories
from utils import sort

# TODO conference badges

config = parse_yaml("config.yaml")
f = open(config["filename"], "w")


# Introduction ########################################################
f.write(h1(config["title"]))
for badge in config["badge"]:
    f.write(badge)
    newline(f)

newline(f)
f.write(config["description"])
newline(f, iter=2)


# Table of Contents ###################################################
f.write(h2("Table of Contents"))
table_of_contents = parse_yaml("data.yaml")
default_level = 1
max_level = config.get("max_level", default_level)
level = default_level
for tol in table_of_contents:
    li(f, a([
        tol["name"],
        config["url"] + "#" + name2link(tol["name"]),
    ]))

    # Deeper levels in table of contents
    while True:
        if level < max_level:
            level += 1
            sub_table_of_contents = find_subdirectories(name2dir(tol["name"]))
            for s in sub_table_of_contents:
                lili(f, a([
                    dir2name(s.name),
                    config["url"] + "/tree/master/" + str(s),
                ]))
        else:
            level = default_level
            break

newline(f)


# Main Content ########################################################
for tol in table_of_contents:
    f.write(h2(tol["name"]))

    datafile = Path(name2dir(tol["name"]))
    if not datafile.is_dir():
        print(f"You must create directory for {tol['name']} and populate it with data.yaml, config.yaml and renderer.py files.")
        continue

    data = parse_yaml(datafile / "data.yaml")
    config_local = parse_yaml(datafile / "config.yaml")

    # Section description
    description = config_local.get("description", None)
    if description is not None:
        f.write(p(description))
        newline(f)

    # Sort content of section
    sort_key = config_local.get("sort_key", None)
    data = sort(data, sort_key, config_local.get("sort_reverse", False))

    exec(f"from {datafile}.renderer import renderer")

    try:
        exec(f"from {datafile}.renderer import renderer_subdir")
        # e.g. content of Papers / README.md
        fp_sub2 = open(str(Path(tol["name"]) / "README.md"), "w")
        fp_sub2.write(h1(tol["name"]))
        fp_sub2.write(a(["Back to awesome edge machine learning", config["url"]]))
        newline(fp_sub2, iter=2)
    except:
        pass

    if not isinstance(data, list):
        data = [data]
    for d in data:
        renderer(f, d, config)

    subdirs = find_subdirectories(datafile)
    for idx, sub in enumerate(subdirs):
        # e.g. content of Papers / AutoML / README.md
        data_sub = parse_yaml(sub / "data.yaml")
        config_sub = parse_yaml(sub / "config.yaml")
        fp_sub = open(sub / "README.md", "w")

        fp_sub.write(h1(dir2name(sub.name)))
        fp_sub.write(a(["Back to awesome edge machine learning", config["url"]]))
        newline(fp_sub, iter=2)
        fp_sub.write(a([f"Back to {datafile}", config["url"] + f"/tree/master/{datafile}"]))
        newline(fp_sub, iter=2)
        fp_sub.write(data[idx]["description"])
        newline(fp_sub, iter=2)

        exec(f"from {str(sub).replace('/', '.')}.renderer import renderer")

        try:
            fp_sub2.write(h2(dir2name(sub.name)))
            newline(fp_sub2)
            fp_sub2.write((data[idx]["description"]))
            newline(fp_sub2)
        except:
            pass

        if config_sub is not None:
            sort_key = config_sub.get("sort_key", None)
            data_sub = sort(data_sub, sort_key, config_sub.get("sort_reverse", False))
            for d in data_sub:
                renderer(fp_sub, d, config)
                try:
                    renderer_subdir(fp_sub2, d, config)
                except:
                    pass
        fp_sub.close()
