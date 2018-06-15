import csv
import numpy as np

def createSubmissionFile(labels, outputFile='submission.csv'):
    with open(outputFile, 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Id", "Prediction"])

        for i, label in enumerate(labels):
            writer.writerow([i+1, label+1])