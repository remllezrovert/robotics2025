outputFile = open("./output_grid_processed.gcode", 'wt')
penDown = "0"
penUp = "10"
speed = "500"
scale = 0.1
def processGcode():
    with open("./output_grid.gcode", 'r') as inputFile:
        for line in inputFile.readlines():
            if line.strip():
                if ("G1" in line or "G0" in line) and ("X" in line or "Y" in line):
                    lineArr = line.split(" ")
                    outArr = []
                    for word in  lineArr:
                        if "G" in word:
                            outArr.append(word)
                            continue
                        elif "X" in word:
                            outArr.append("X" + str(float(word.replace("X", "")) / 30 + 67.5))
                        elif "Y" in line:
                            outArr.append("Y" + str(float(word.replace("Y", "")) / 30))
                        else:
                            outArr.append(word)
                    line = " ".join(outArr)    
                    outputFile.write(line + "\n")
                    print(line)
                    pass
                if "laser" in line or "power" in line:
                    pass   
                elif "S300" in line:
                    pass
                elif "M3" in line:
                    outputFile.write(line.replace("M3", "G0 Z" + penDown))
                elif "M5" in line:
                    outputFile.write(line.replace("M5","G0 Z" + penUp))
                elif "rapid_move:" in line:
                    outputFile.write(";      rapid_move: " + speed + ",\n")
                else:
                    outputFile.write(line)
        

## m3 means laser on, replace this with "pen down"
## m5 means laser off, replace this with "pen up"
## Adust speed everywhere
## ensure that the pen height has a baseline raised height just above the choolate bar

processGcode()