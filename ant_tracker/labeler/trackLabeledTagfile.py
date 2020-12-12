from classes import AntCollection

filename = "dia"
tagfile = f"{filename}.tag"
trackerJson = AntCollection.deserialize(filename=tagfile).serializeAsTracker()
with open("./dia-labeled2.rtg","w") as target:
    target.write(trackerJson)
