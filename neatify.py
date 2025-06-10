import csv

with open('infer_sentiments.csv', 'rt') as fd, open('infer_neat.csv', 'wt') as fd_out:
    reader = csv.reader(fd)
    writer = csv.writer(fd_out)
    next(reader)
    writer.writerows(
        list(cell.strip().replace('​', '').replace('﻿', '') for cell in row if cell) for row in reader
    )
