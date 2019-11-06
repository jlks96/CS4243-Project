import os
from pathlib import Path

characters = ["waldo", "wenda", "wizard"]
components = ["torso", "head"]
printoutput = ""
source_dir = os.path.join("..", "datasets", "PositiveSamples")
for character in characters:
    for component in components: 
        source = os.path.join(source_dir, character, component)
        total = 0
        for path, subdirs, files in os.walk(source):
            for subdir in subdirs:
                newsource = os.path.join(source, subdir)
                for path2, subdirs2, files2 in os.walk(newsource): 
                    print(character, component, subdir, len(files2))
                    total += len(files2)
        printoutput += (character + " " + component + " " + str(total) + "\n")

print("\n\n" + printoutput)