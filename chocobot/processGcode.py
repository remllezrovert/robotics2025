outputFile = open("./output_grid_processed.gcode", 'wt')
penDown = "0"
penUp = "10"
speed = "50"
def processGcode():
    with open("./output_grid.gcode", 'r') as inputFile:
        for line in inputFile.readlines():
            if line.strip():
                if "laser" in line or "power" in line:
                    pass   
                elif "S300" in line:
                    pass
                elif "M3" in line:
                    outputFile.write(line.replace("M3", "G0 Z" + penDown))
                elif "M5" in line:
                    outputFile.write(line.replace("M5","G0 Z" + penUp))
                elif "rapid_move:" in line:
                    outputFile.write(line.replace(";      rapid_move: 60,",";      rapid_move: " + speed + ","))
                else:
                    outputFile.write(line)
        

## m3 means laser on, replace this with "pen down"
## m5 means laser off, replace this with "pen up"
## Adust speed everywhere
## ensure that the pen height has a baseline raised height just above the choolate bar

processGcode()