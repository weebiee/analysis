import asyncio
import csv
import itertools
from argparse import ArgumentParser
from pathlib import Path

import grpc.aio
import numpy as np
import tqdm

SENTIMENTS = ['积极', '消极', '中性']


def get_sentiment(scores) -> str:
    return SENTIMENTS[np.argmax([scores.positivity, scores.negativity, scores.neutrality])]


async def main():
    parser = ArgumentParser()
    parser.add_argument('node', type=str)
    parser.add_argument('--secure', '-S', action='store_true', default=False)
    args = parser.parse_args()

    channel = grpc.aio.secure_channel(args.node, grpc.ssl_channel_credentials()) \
        if args.secure else grpc.aio.insecure_channel(args.node)

    import Evaluator_pb2_grpc as _rpc
    import Evaluator_pb2 as _pb
    stub = _rpc.EvaluatorStub(channel)

    source, out = Path('infer_neat.csv'), Path('infer_results.csv')

    completed_lines = 0
    total_lines = 0
    if not out.exists():
        out.touch()

    with open(out, 'rt') as fd_in:
        while fd_in.readline():
            completed_lines += 1

    with open(source, 'rt') as fd:
        while fd.readline():
            total_lines += 1

    with open(source, 'rt') as fd_in, open(out, 'at') as fd_out, tqdm.tqdm(total=total_lines) as pbar:
        reader = csv.reader(fd_in)
        for _ in range(completed_lines):
            next(reader)

        if completed_lines > 0:
            print(f'resuming from line {completed_lines}')

        writer = csv.writer(fd_out)
        for batch in itertools.batched(itertools.chain(reader), 50):
            rows = list(batch)
            res = await stub.GetScores(_pb.GetScoresRequest(
                phrases=list(r[2] for r in rows)
            ))
            if not res.ok:
                raise RuntimeError(f'batch failed to process: {res.err_msg}')
            scores = list(res.scores)
            writer.writerows(r + [get_sentiment(scores[idx])] for idx, r in enumerate(rows))

            fd_out.flush()
            pbar.update(len(rows))


if __name__ == '__main__':
    asyncio.run(main())
